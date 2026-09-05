// Prototype alternate harness entry point.
//
// Same Exo tools and instructions as the default "exo" harness
// (exo/harness.ts), but the turn loop is driven by a real
// @earendil-works/pi-agent-core Agent instead of Exo's own hand-written
// runResponsesTurnLoop (exoharness/typescript/model-runtime/turn-loop.ts).
//
// Defaults to file tools on (write/edit/read - see tools/file-tools.ts) as of
// the benchmarking below; the shell-only variant that used to live here is
// preserved as harness-pi-core-shell-only.ts for comparison, not deleted.
// Across every model benchmarked, file tools matched or beat shell-only, most
// dramatically on weaker/faster models that struggle with shell quoting:
//   gpt-5-nano:            27/27 vs 26/27
//   Gemini 2.5 Flash Lite: 23/27 vs 14/27
// No case favored shell-only. This module's own history is the reason: a
// runaway tool-call loop that cost ~$9 of credit traced back to a shell
// heredoc quoting collapse (see git log for the incident and its fixes).
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
  MAX_CONSECUTIVE_TOOL_ERRORS,
  materializePromptMessages,
  type HarnessToolRegistry,
  type TurnContext,
} from "@exo/harness";
import { runtimeFromModelBinding } from "@exo/model-runtime/responses";
import { appendEvents, resolveLlmBinding } from "@exo/model-runtime/shared";

import { exoInstructions, registerExoTools } from "./harness";
import {
  buildModelStub,
  createExoStreamFn,
  createProtectedPathBeforeToolCallHook,
  exoMessagesToAgentSeed,
  looksLikeUnfinishedTurn,
  piEventToExoEvents,
  toolInstanceToAgentTool,
  type PiRecordableEvent,
} from "./tools/pi-agent-adapters";

// Cap on how many times a single turn will nudge the model to retry an
// unfinished turn (malformed tool call, or a genuinely empty response)
// before giving up and letting the turn end anyway - see
// looksLikeUnfinishedTurn's comment for why this exists. Chosen to
// comfortably cover what live benchmarking needed (usually 1, at most a
// couple) without risking a runaway loop against a model that never manages
// a real tool call.
const MAX_UNFINISHED_TURN_NUDGES = 5;

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context);
  },
});

export interface PiCoreTurnOptions {
  // Defaults to registerExoTools, which - since file tools were promoted to
  // the practical profile's real tool set (see git history) - already
  // includes write/edit/read. This used to be a `fileTools?: boolean` flag
  // that called registerFileTools a second time on top of registerExoTools;
  // once registerExoTools started including it too, that became a literal
  // double registration ("tool is already registered: write"), reproduced
  // live: every pi-core-files-ds benchmark call failed instantly, 0 rounds,
  // no error event, because the harness process crashed before ever
  // materializing a prompt. Pass a custom registerTools (see
  // harness-pi-core-shell-only.ts) to get a tool surface other than the
  // real default - not a flag layered on top of it.
  registerTools?: (
    tools: HarnessToolRegistry,
    context: TurnContext,
  ) => Promise<void> | void;
}

export async function runPiCoreTurn(
  context: TurnContext,
  options: PiCoreTurnOptions = {},
): Promise<void> {
  const tools = createToolRegistry(context);
  await (options.registerTools ?? registerExoTools)(tools, context);

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

  // Config-audit findings (both silent gaps against the default harness,
  // neither surfaced by benchmarking so far):
  //
  //   - maxToolRoundTrips: the original turn loop enforces
  //     context.agentConfig.maxToolRoundTrips as a hard cap
  //     (exoharness/typescript/model-runtime/turn-loop.ts). Agent has no
  //     such option directly - shouldStopAfterTurn is the documented way to
  //     add one. Without it, an operator-configured round limit (a real
  //     cost/safety control, set via --max-tool-round-trips) was silently
  //     unenforced here. completedRounds counts turns *after* they've
  //     already run (shouldStopAfterTurn fires post-turn_end), so the exact
  //     round arithmetic doesn't match turn-loop.ts's pre-round check
  //     bit-for-bit - the point is having a real cap, not replicating an
  //     off-by-one.
  //
  //   - toolExecution: Agent defaults to "parallel" (agent.js). The
  //     original loop executes tool calls strictly sequentially
  //     (`for (const toolCall of toolCalls) { await ... }`), which matters
  //     here because Exo's tools (shell chief among them) share one sandbox
  //     filesystem - two tool calls from the same assistant message running
  //     concurrently could race on the same files. Left at the pi-agent-core
  //     default, this harness would silently risk that race the first time
  //     a model asked for more than one tool call in a single message.
  const maxToolRoundTrips = context.agentConfig.maxToolRoundTrips;
  let completedRounds = 0;
  // A round-count cap alone can't tell a task that legitimately needs many
  // tool calls apart from one stuck resending the same failing call - both
  // just look like "many rounds". The real-world case that motivated this:
  // Qwen3 Coder Flash sent a malformed shell-tool argument, got a
  // validation error, and resent the exact same call unchanged for 230
  // consecutive rounds over 2.5 hours before anyone noticed. Tracking a
  // consecutive-failure streak catches that in a handful of rounds
  // regardless of what maxToolRoundTrips is set to, without penalizing a
  // turn that is making real (even if slow) progress.
  let consecutiveToolErrors = 0;

  const agent = new Agent({
    initialState: {
      systemPrompt: seed.systemPrompt,
      model,
      tools: agentTools,
      messages: seed.messages,
    },
    streamFn: createExoStreamFn(runtime, modelBinding.model, {
      onFirstChunk: (ttftMs) => context.stream.firstChunk(ttftMs),
      onTextDelta: (text) => context.stream.text(text),
    }),
    beforeToolCall: createProtectedPathBeforeToolCallHook(),
    toolExecution: "sequential",
    shouldStopAfterTurn: async (turnContext) => {
      completedRounds += 1;
      if (
        turnContext.toolResults.length > 0 &&
        turnContext.toolResults.every((result) => result.isError)
      ) {
        consecutiveToolErrors += 1;
      } else {
        consecutiveToolErrors = 0;
      }
      if (consecutiveToolErrors >= MAX_CONSECUTIVE_TOOL_ERRORS) {
        try {
          await context.exoharness.current.turn.writeArtifactText({
            path: "pi-core/aborted-consecutive-tool-errors.json",
            text: JSON.stringify({
              consecutiveToolErrors,
              completedRounds,
              at: new Date().toISOString(),
            }),
          });
        } catch {
          // Instrumentation is best-effort; the abort itself must not depend on it.
        }
        return true;
      }
      return (
        maxToolRoundTrips !== null &&
        maxToolRoundTrips !== undefined &&
        completedRounds > maxToolRoundTrips
      );
    },
  });

  let unfinishedTurnNudges = 0;
  agent.subscribe(async (event) => {
    if (isRecordableEvent(event)) {
      const events = piEventToExoEvents(event);
      if (events.length > 0) {
        await appendEvents(context, events);
      }
    }
    // A turn that ended with no tool results and either a botched tool-call
    // attempt or a genuinely empty response isn't actually finished - nudge
    // a retry instead of letting the Agent treat it as a normal stop.
    // followUp() only takes effect once the agent would otherwise stop
    // (pi-agent-core's own mechanism for this - pi-coding-agent's CLI relies
    // on the same "hasMoreToolCalls || pendingMessages" loop condition, it
    // just tends to get well-formed, non-empty responses from the model
    // more often in the first place).
    if (
      event.type === "turn_end" &&
      event.toolResults.length === 0 &&
      unfinishedTurnNudges < MAX_UNFINISHED_TURN_NUDGES &&
      looksLikeUnfinishedTurn(event.message)
    ) {
      unfinishedTurnNudges += 1;
      // Instrumentation, not behaviour: one artifact per nudge so a benchmark
      // can count how often this recovery path actually fires (jq over
      // artifact_written events whose path starts with "pi-core/nudge-").
      // Whether this hand-written nudge is still earning its place once
      // file-tools remove the main source of botched turns is an empirical
      // question, and it was previously unmeasurable. Failing to record must
      // never cost a turn, hence the swallowed error.
      try {
        await context.exoharness.current.turn.writeArtifactText({
          path: `pi-core/nudge-${unfinishedTurnNudges}.json`,
          text: JSON.stringify({
            nudge: unfinishedTurnNudges,
            reason: "unfinished-turn",
            at: new Date().toISOString(),
          }),
        });
      } catch {
        // Instrumentation is best-effort.
      }
      agent.followUp({
        role: "user",
        content:
          "Your previous reply didn't finish the task - it was either empty or described a tool call as plain text instead of actually invoking it. Continue the task: call the tool using the actual function-calling mechanism, or give a real final answer if the work is genuinely done.",
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
