// Prototype alternate harness entry point.
//
// Same Exo tools and instructions as the default "exo" harness
// (exo/harness.ts), but the turn loop is driven by a real
// @earendil-works/pi-agent-core Agent instead of Exo's own hand-written
// runResponsesTurnLoop (exoharness/typescript/model-runtime/turn-loop.ts).
//
// This does NOT replace the default harness or the "practical" profile -
// point --harness at this module's path to try it instead:
//   ./target/debug/exo agent create "Pi-core test" --harness exo/harness-pi-core.ts --model ...
// (see exoharness/docs/coding-agent-harnesses.md for how --harness resolves
// a TypeScript module path).
//
// What stays exactly as it is today: the Rust event log (writes still go
// through the same appendEvents -> context.exoharness.current.turn.addEvents
// RPC used by every other harness), sandbox orchestration, secret store,
// scheduler, guardian. Only the in-turn model+tool loop is replaced.
//
// See exo/tools/pi-agent-adapters.ts for the two integration seams this
// prototype validates (AgentTool adapter, StreamFn adapter) and the list of
// known limitations (no incremental streaming, best-effort historical
// message replay, best-effort usage accounting) - none of those affect the
// Rust core or the event log's durability guarantees, only the fidelity of
// what gets recorded from this particular harness.

import { Agent } from "@earendil-works/pi-agent-core";

import {
  createToolRegistry,
  defineHarness,
  materializePromptMessages,
  type TurnContext,
} from "@exo/harness";
import { runtimeFromModelBinding } from "@exo/model-runtime/responses";
import { appendEvents, resolveLlmBinding } from "@exo/model-runtime/shared";

import { exoInstructions, registerExoTools } from "./harness";
import {
  buildModelStub,
  createExoStreamFn,
  exoMessagesToAgentSeed,
  looksLikeMalformedToolCallAttempt,
  piEventToExoEvents,
  toolInstanceToAgentTool,
  type PiRecordableEvent,
} from "./tools/pi-agent-adapters";

// Cap on how many times a single turn will nudge the model to retry a
// malformed tool call before giving up and letting the turn end anyway - see
// looksLikeMalformedToolCallAttempt's comment for why this exists. Chosen to
// comfortably cover what live benchmarking needed (usually 1, at most a
// couple) without risking a runaway loop against a model that never manages
// a real tool call.
const MAX_MALFORMED_TOOL_CALL_NUDGES = 5;

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context);
  },
});

async function runPiCoreTurn(context: TurnContext): Promise<void> {
  const tools = createToolRegistry(context);
  await registerExoTools(tools, context);

  const instructions = await exoInstructions(context, tools);
  const materialized = await materializePromptMessages(
    context.exoharness.current.conversation,
    instructions,
  );
  const seed = exoMessagesToAgentSeed(materialized);
  if (seed.messages.length === 0) {
    throw new Error(
      "pi-core harness needs at least one prior user turn to continue from " +
        "(materialized conversation was empty) - this prototype does not " +
        "yet support turns with no user message, unlike the default harness",
    );
  }

  const modelBinding = await resolveLlmBinding(context);
  const runtime = runtimeFromModelBinding(context.agentConfig, modelBinding);
  const model = buildModelStub(modelBinding.model);
  const agentTools = tools
    .instances()
    .map((tool) => toolInstanceToAgentTool(tool, context));

  const agent = new Agent({
    initialState: {
      systemPrompt: seed.systemPrompt,
      model,
      tools: agentTools,
      messages: seed.messages,
    },
    streamFn: createExoStreamFn(runtime, modelBinding.model),
  });

  let malformedToolCallNudges = 0;
  agent.subscribe(async (event) => {
    if (isRecordableEvent(event)) {
      const events = piEventToExoEvents(event);
      if (events.length > 0) {
        await appendEvents(context, events);
      }
    }
    // A turn that ended with no tool results but text that looks like a
    // botched tool-call attempt isn't actually finished - nudge a retry
    // instead of letting the Agent treat it as a normal stop. followUp()
    // only takes effect once the agent would otherwise stop (pi-agent-core's
    // own mechanism for this - pi-coding-agent's CLI relies on the same
    // "hasMoreToolCalls || pendingMessages" loop condition, it just tends to
    // get well-formed tool calls from the model more often in the first
    // place).
    if (
      event.type === "turn_end" &&
      event.toolResults.length === 0 &&
      malformedToolCallNudges < MAX_MALFORMED_TOOL_CALL_NUDGES &&
      looksLikeMalformedToolCallAttempt(event.message)
    ) {
      malformedToolCallNudges += 1;
      agent.followUp({
        role: "user",
        content:
          "Your previous reply did not contain a real tool call - it looks like you described one as plain text instead of actually invoking the tool. Call the tool again using the actual function-calling mechanism, not text describing it.",
        timestamp: Date.now(),
      });
    }
  });

  // seed.messages already ends with the latest user turn (materialized from
  // the conversation as of right now), so continue() rather than prompt() -
  // prompt() would append a second copy of that same turn.
  await agent.continue();
}

function isRecordableEvent(event: {
  type: string;
}): event is PiRecordableEvent {
  return (
    event.type === "tool_execution_start" ||
    event.type === "tool_execution_end" ||
    event.type === "message_end"
  );
}
