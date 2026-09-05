import {
  createToolRegistry,
  looksLikeUnfinishedTurn,
  materializePromptMessages,
  messagesEvent,
  MAX_CONSECUTIVE_TOOL_ERRORS,
  registerBuiltInTools,
  registerInstalledTools,
  registerLegacyAgentToolsFromDirectoryIfExists,
  registerLibraryToolModulePath,
  toolResultEventIsError,
  turnMetadata,
  userTextMessage,
  type BuiltInToolName,
  type EventData,
  type HarnessToolRegistry,
  type Message,
  type TurnContext,
} from "@exo/harness";
import {
  responseMessages,
  responseToLinguaEvents,
  responseToolCalls,
  runtimeFromModelBinding,
  type NativeResponsesRequest,
  type ResponsesRuntimeLike,
  type TraceParent,
} from "@exo/model-runtime/responses";
import { ensureTable } from "@exo/model-runtime/cost";

import { resolveLlmBinding } from "./shared";

// Cap on how many times a single turn will nudge the model to retry an
// unfinished turn (empty response, or a tool call described as plain text)
// before giving up and letting the turn end anyway. Ported from
// exo/harness-pi-core.ts's identical constant/mechanism - ~$0 without one,
// a hard tier-6 benchmark task showed gpt-5-nano producing this exact
// failure shape on exo/harness.ts (the harness every production "exo" agent
// uses) with no recovery and nothing in the event log explaining why the
// turn ended empty.
const MAX_UNFINISHED_TURN_NUDGES = 5;
const UNFINISHED_TURN_NUDGE_TEXT =
  "Your previous reply didn't finish the task - it was either empty or described a tool call as plain text instead of actually invoking it. Continue the task: call the tool using the actual function-calling mechanism, or give a real final answer if the work is genuinely done.";

export interface ResponsesTurnLoopOptions {
  instructions?: (
    context: TurnContext,
    tools: HarnessToolRegistry,
  ) => Message[] | Promise<Message[]>;
  registerTools?: (
    tools: HarnessToolRegistry,
    context: TurnContext,
  ) => Promise<void> | void;
}

export async function runResponsesHarnessTurn(
  context: TurnContext,
  options: ResponsesTurnLoopOptions = {},
): Promise<void> {
  await ensureTable(); // load the price table once so cost is ready when events are built
  const modelBinding = await resolveLlmBinding(context);
  const runtime = runtimeFromModelBinding(context.agentConfig, modelBinding);
  await runtime.runTurn(context, (turnParent) =>
    runResponsesTurnLoop(
      runtime,
      context,
      turnParent,
      modelBinding.model,
      options,
    ),
  );
}

export async function createDefaultToolRegistry(
  context: TurnContext,
  builtInToolNames: BuiltInToolName[] = defaultBuiltInToolNames(context),
): Promise<HarnessToolRegistry> {
  const tools = createToolRegistry(context);
  registerBuiltInTools(tools, context, builtInToolNames);
  for (const modulePath of context.agentConfig.typescript?.toolModulePaths ??
    []) {
    await registerLibraryToolModulePath(tools, context, modulePath);
  }
  await registerConfiguredAgentTools(tools, context);
  return tools;
}

export async function registerConfiguredAgentTools(
  tools: HarnessToolRegistry,
  context: TurnContext,
): Promise<void> {
  await registerInstalledTools(tools, context);
  if (context.agentConfig.enableAgentToolCreation) {
    await registerLegacyAgentToolsFromDirectoryIfExists(tools, context);
  }
}

export function defaultBuiltInToolNames(
  context: TurnContext,
): BuiltInToolName[] {
  const names: BuiltInToolName[] = ["shell"];
  if (context.agentConfig.enableAgentToolCreation) {
    names.push("install_agent_tool", "uninstall_agent_tool");
  }
  return names;
}

export function basicHarnessInstructions(context: TurnContext): Message[] {
  return context.agentConfig.enableAgentToolCreation
    ? [...context.agentConfig.instructions, agentToolCreationInstruction()]
    : context.agentConfig.instructions;
}

export function agentToolCreationInstruction(): Message {
  return {
    role: "developer",
    content:
      "Agent-created tools are supported. When the user asks you to create a reusable tool, call install_agent_tool with a complete TypeScript moduleSource. Do not claim the tool was created unless install_agent_tool returns ok: true. The moduleSource must use type-only imports from @exo/harness/tool and default-export a Tool using { definition, initializationParameters, initialize(...) } satisfies Tool; definition.parameters must be a strict JSON schema object with additionalProperties: false; handlers must implement execute(args, execution), not invoke or call. Do not use zod, inputSchema, external npm packages, or runtime imports from @exo/harness/tool. After install_agent_tool succeeds, the new tool is available in the next model round of the same turn, so use it directly rather than falling back to shell. Use uninstall_agent_tool to remove an agent-created tool that is obsolete or conflicts with another tool name.",
  };
}

async function runResponsesTurnLoop(
  runtime: ResponsesRuntimeLike,
  context: TurnContext,
  turnParent: TraceParent,
  model: string,
  options: ResponsesTurnLoopOptions,
): Promise<string | null> {
  const { conversation } = context.exoharness.current;
  const maxToolRoundTrips = context.agentConfig.maxToolRoundTrips;
  let latestEventId: string | null = null;
  let consecutiveToolErrors = 0;
  let unfinishedTurnNudges = 0;

  for (let round = 0; ; round += 1) {
    if (
      maxToolRoundTrips !== null &&
      maxToolRoundTrips !== undefined &&
      round > maxToolRoundTrips
    ) {
      return latestEventId;
    }
    if (consecutiveToolErrors >= MAX_CONSECUTIVE_TOOL_ERRORS) {
      try {
        await context.exoharness.current.turn.writeArtifactText({
          path: "turn-loop/aborted-consecutive-tool-errors.json",
          text: JSON.stringify({
            consecutiveToolErrors,
            round,
            at: new Date().toISOString(),
          }),
        });
      } catch {
        // Instrumentation is best-effort; the abort itself must not depend on it.
      }
      return latestEventId;
    }

    const tools = options.registerTools
      ? createToolRegistry(context)
      : await createDefaultToolRegistry(context);
    if (options.registerTools) {
      await options.registerTools(tools, context);
    }
    const messages = await materializePromptMessages(
      conversation,
      options.instructions
        ? await options.instructions(context, tools)
        : basicHarnessInstructions(context),
    );
    const request: NativeResponsesRequest = {
      model,
      messages,
      tools: tools.definitions(),
      maxOutputTokens: context.agentConfig.maxOutputTokens,
      metadata: turnMetadata(context),
    };

    const response = context.streaming
      ? await runtime.completeStream(
          request,
          {
            onFirstChunk: (ttftMs) => context.stream.firstChunk(ttftMs),
            onTextDelta: (text) => context.stream.text(text),
          },
          {
            parent: turnParent,
            roundIndex: round,
          },
        )
      : await runtime.complete(request, {
          parent: turnParent,
          roundIndex: round,
        });

    const events = responseToLinguaEvents(response);
    if (events.length > 0) {
      latestEventId = await appendTurnEvents(context, events);
    }

    const toolCalls = responseToolCalls(response);
    const hasSyntheticToolResult = events.some(
      (event) => event.type === "tool_result",
    );
    if (toolCalls.length === 0) {
      if (hasSyntheticToolResult) {
        continue;
      }
      // A model call that produces no assistant message at all (gpt-5-nano
      // returning a bare reasoning item with no message/output_text - the
      // shape actually observed live) is the same failure as an assistant
      // message with empty content, not a reason to skip the nudge: without
      // this, `lastMessage` is undefined and the check below never fires.
      const lastMessage = responseMessages(response).at(-1);
      const unfinished = !lastMessage || looksLikeUnfinishedTurn(lastMessage);
      if (unfinished && unfinishedTurnNudges < MAX_UNFINISHED_TURN_NUDGES) {
        unfinishedTurnNudges += 1;
        // Instrumentation, not behaviour: lets a benchmark or operator count
        // how often this recovery path actually fires (jq over
        // artifact_written events whose path starts with "turn-loop/nudge-").
        // Failing to record must never cost a turn, hence the swallowed error.
        try {
          await context.exoharness.current.turn.writeArtifactText({
            path: `turn-loop/nudge-${unfinishedTurnNudges}.json`,
            text: JSON.stringify({
              nudge: unfinishedTurnNudges,
              reason: "unfinished-turn",
              at: new Date().toISOString(),
            }),
          });
        } catch {
          // Instrumentation is best-effort.
        }
        latestEventId = await appendTurnEvents(context, [
          messagesEvent([userTextMessage(UNFINISHED_TURN_NUDGE_TEXT)]),
        ]);
        continue;
      }
      return latestEventId;
    }

    for (const toolCall of toolCalls) {
      const toolResultEvents = await runtime.traceToolCall(
        turnParent,
        context,
        toolCall,
        round,
        (toolCall) => tools.executePending([toolCall]),
      );
      if (toolResultEvents.length > 0) {
        latestEventId = await appendTurnEvents(context, toolResultEvents);
      }
      if (toolResultEvents.some(toolResultEventIsError)) {
        consecutiveToolErrors += 1;
      } else if (toolResultEvents.length > 0) {
        consecutiveToolErrors = 0;
      }
    }
  }
}

async function appendTurnEvents(
  context: TurnContext,
  data: EventData[],
): Promise<string> {
  return (await context.exoharness.current.turn.addEvents(data)).latestEventId;
}
