// Prototype adapters bridging Exo's own tool/model abstractions to the real
// @earendil-works/pi-agent-core Agent engine. See ../harness-pi-core.ts for
// the harness entry point that wires these together into a turn.
//
// This exists to validate (with real, installed packages, not guesses) the
// two integration seams identified while comparing Exo's turn loop
// (exoharness/typescript/model-runtime/turn-loop.ts) against Pi's real
// architecture:
//
//   - AgentTool: pi's Tool.parameters wants a TypeBox `TSchema`, but a
//     TypeBox schema is a plain JSON Schema object at runtime - Exo's
//     already-strict JSON Schema just needs a type-level cast, not a
//     rewrite. execute() throws on failure (pi's convention) instead of
//     returning { ok: false } (Exo's convention); adapted here.
//   - StreamFn (pi-agent-core/dist/stream-fn.d.ts): must never throw.
//     Failures must be encoded as a start+error event pair ending in an
//     AssistantMessage with stopReason "error"/"aborted".
//
// Known prototype limitations (not yet production-grade):
//   - Historical messages seeded into a fresh Agent (exoMessagesToAgentSeed,
//     used once at turn start) are re-encoded as plain text rather than a
//     provider-native transcript replay - fine for coherent context on the
//     first round. Mid-turn round-tripping (piMessageToExoMessage, used on
//     every round after the first) preserves real tool_call/tool_result
//     content blocks via Exo's own toolResultMessage() helper and message
//     content-part convention, matching what materializePromptMessages
//     produces from a live event log - this was the fix for a real bug
//     (see below), not a remaining gap.
//   - No real multimodal support: piContentText always flattens images to
//     the literal string "[image]" (matching Exo's own contentText()
//     convention so the loss is at least visible, not silent - see its own
//     comment). An image a user attaches never actually reaches the model
//     through this harness. The default harness's fidelity here is
//     unverified too; flagged as a gap either way, not claimed as a
//     regression.
//   - Usage accounting through this path is best-effort (falls back to 0);
//     Exo's authoritative cost accounting happens elsewhere and is untouched.
//
// AgentOptions fields deliberately left unset, decided by config audit
// (2026-09-04) rather than left unconsidered:
//   - afterToolCall: not needed as a separate hook - compactToolResultForModel
//     (large-result truncation, see toolInstanceToAgentTool) already runs at
//     the one place that needs it, inside the tool's own execute(), which is
//     simpler than routing the same data through a second Agent-level hook.
//   - transformContext: this is pi-agent-core's documented seam for custom
//     context pruning, and the closest thing to compaction available at this
//     integration depth (see harness-pi-core.ts's header for why real
//     compact()/AgentHarness isn't). Left unimplemented because no run in
//     this session has actually hit a context-overflow failure to design
//     against - compactToolResultForModel now also caps the largest known
//     source of unbounded growth. Revisit if a real overflow shows up.
//   - getApiKey, onPayload, onResponse, sessionId, thinkingBudgets,
//     transport, maxRetryDelayMs: these decorate pi-ai's own provider/retry
//     dispatch (models.streamSimple and friends). createExoStreamFn bypasses
//     that dispatch entirely and calls Exo's own runtime.completeStream()
//     directly, so none of these hooks would ever fire - not gaps, just not
//     applicable to this streamFn's architecture.
//
// Bug fixed after live benchmarking (2026-09-04): piMessageToExoMessage
// originally flattened every pi message to plain text, including assistant
// tool calls. That produced a message sequence a "tool" role message can't
// legally follow (no matching tool_call to point back at), which made the
// *second* runtime.complete() call of a turn fail - silently, because
// createExoStreamFn's never-throw contract turns that failure into an empty
// error AssistantMessage, and piEventToExoEvents only records message_end
// events that carry text. Net effect: any task needing more than one tool
// round-trip stopped dead after the first tool call, with no error visible
// anywhere in Exo's event log. Confirmed via exo conversation send against a
// live sandbox + free OpenRouter model, comparing against the original
// harness on identical prompts.

import type {
  AgentTool,
  AgentToolResult,
  BeforeToolCallContext,
  BeforeToolCallResult,
  StreamFn,
} from "@earendil-works/pi-agent-core";
import {
  createAssistantMessageEventStream,
  type Api as PiApi,
  type AssistantMessage,
  type Context as PiContext,
  type ImageContent as PiImageContent,
  type Message as PiMessage,
  type Model as PiModel,
  type TextContent as PiTextContent,
} from "@earendil-works/pi-ai";
import type { TSchema } from "typebox";

import {
  assistantTextMessage,
  messageText,
  buildTruncatedPreview,
  messagesEvent,
  toolRequestedEvent,
  toolResultEvent,
  toolResultMessage,
  type EventData,
  type JsonObject,
  type JsonValue,
  type Message,
  type ToolInstance,
  type TurnContext,
} from "@exo/harness";
import {
  responseMessages,
  responseToolCalls,
  type NativeResponsesRequest,
  type ResponsesRuntimeLike,
} from "@exo/model-runtime/responses";
import type { Response as OpenAIResponse } from "openai/resources/responses/responses";

// Copied from pi-agent-core's own agent.js EMPTY_USAGE default so a
// synthetic AssistantMessage satisfies the (non-optional) Usage shape.
const EMPTY_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

// ---------------------------------------------------------------------------
// Tool adapter: ToolInstance -> AgentTool
// ---------------------------------------------------------------------------

export function toolInstanceToAgentTool(
  tool: ToolInstance,
  context: TurnContext,
): AgentTool {
  return {
    name: tool.definition.name,
    label: tool.definition.name,
    description: tool.definition.description,
    parameters: tool.definition.parameters as unknown as TSchema,
    async execute(toolCallId, params): Promise<AgentToolResult<unknown>> {
      const result = await tool.handler.execute(params as JsonObject, {
        context,
        toolCallId,
      });
      if (isToolFailure(result)) {
        throw new Error(toolFailureMessage(result));
      }
      const compacted = await compactToolResultForModel(
        context,
        tool.definition.name,
        toolCallId,
        result,
      );
      return {
        content: [{ type: "text", text: compacted.text }],
        details: compacted.details,
      };
    },
  };
}

// Mirrors exoharness/typescript/harness/tools.ts's compactToolResult: large
// tool output goes to an artifact, not straight into the model's context,
// with a short preview standing in. This adapter had no such limit until
// found by inspection (not benchmarking - the free models used so far never
// happened to produce a giant tool result) - a tool like shell running `cat`
// on a large file would otherwise put the *entire* output into every
// subsequent round's context for the rest of the turn, via
// piMessageToExoMessage carrying AgentToolResult.details through unbounded.
// Not a byte-for-byte port: the original also splits shell stdout/stderr
// into their own artifacts and threads artifact references through the
// ToolResult value itself; this is a simpler single-artifact mirror that
// still bounds the size, which is the property that matters here.
const TOOL_RESULT_INLINE_LIMIT_CHARS = 8_000;

async function compactToolResultForModel(
  context: TurnContext,
  toolName: string,
  toolCallId: string,
  result: unknown,
): Promise<{ text: string; details: unknown }> {
  const serialized = stringifyToolResult(result);
  if (serialized.length <= TOOL_RESULT_INLINE_LIMIT_CHARS) {
    return { text: serialized, details: result };
  }
  const artifact = await context.exoharness.current.turn.writeArtifactText({
    path: `tool-results/${sanitizePathSegment(toolName)}/${sanitizePathSegment(toolCallId)}/result.json`,
    text: serialized,
  });
  // The artifact reference alone is worse than nothing here: it reads like a
  // path the model can open, and it is host-side, so every attempt to follow
  // it fails. buildTruncatedPreview keeps both ends of the output and adds the
  // one location that is actually reachable from inside the sandbox.
  const text = buildTruncatedPreview(
    serialized,
    result,
    `full result written to artifact ${artifact.artifactId} at ${artifact.path}`,
  );
  return {
    text,
    details: {
      truncated: true,
      // Must be the same text the model was shown, not a second, narrower
      // slice of its own. details is what piMessageToExoMessage replays into
      // context on every later round, so a head-only preview here silently
      // undid the head+tail fix one round after the tool ran: the tail was in
      // the immediate result and gone from the transcript the model reasoned
      // over. Measured as pi-core answering NOT-VISIBLE while the value it
      // was asked for sat in the tool result it had just received.
      preview: text,
      artifactId: artifact.artifactId,
      path: artifact.path,
      sizeBytes: artifact.sizeBytes,
    },
  };
}

function sanitizePathSegment(value: string): string {
  return value.replace(/[^A-Za-z0-9_.-]+/g, "_").slice(0, 96) || "unknown";
}

function isToolFailure(
  result: unknown,
): result is { ok: false; error?: unknown } {
  return (
    Boolean(result) &&
    typeof result === "object" &&
    !Array.isArray(result) &&
    (result as { ok?: unknown }).ok === false
  );
}

function toolFailureMessage(result: { ok: false; error?: unknown }): string {
  return typeof result.error === "string" ? result.error : "tool call failed";
}

function stringifyToolResult(result: unknown): string {
  return typeof result === "string" ? result : JSON.stringify(result, null, 2);
}

// ---------------------------------------------------------------------------
// beforeToolCall guardrail: blocks shell commands that write to obviously
// protected paths. Same defaults and intent as pi-coding-agent's own
// protected-paths.ts extension (~/.pi/agent/extensions/protected-paths.ts)
// - block outright, headless-safe, no prompt - adapted for Exo's single
// "shell" tool (a raw command string) instead of Pi's separate write/edit
// tools (a structured path argument), so this matches on the command text
// rather than a dedicated path field. A text match is inherently looser
// than a real path argument (it can't tell "rm .env.example" from
// "rm .env", for one) - conservative on purpose, since a false positive
// just costs the model one blocked attempt while a false negative costs a
// real file.
//
// This is a speed bump against a model naively writing where it shouldn't,
// not a security boundary: a plain substring+regex match over the raw
// command text is trivially defeated by anything that keeps ".env" out of
// the literal text the guardrail sees - shell variable indirection
// (`f=.env; rm .$f`... though `f=.env` itself still matches, a cleverer
// split doesn't), quoting/escaping tricks, base64/hex-decoded commands
// piped to a shell, or writing through another tool entirely if one is ever
// registered. It stops the failure mode actually observed (a model asked to
// touch some other file drifting onto .env by mistake), not a deliberately
// adversarial one - same scope pi-coding-agent's own version claims.
// ---------------------------------------------------------------------------

const DEFAULT_PROTECTED_PATHS = [
  ".env",
  ".git/",
  "node_modules/",
  "__pycache__/",
];

// Only block commands that look like they *write* - a bare `cat .env` or
// `grep foo .git/config` is a read, not something this guardrail is for.
const WRITE_LOOKING_COMMAND_PATTERN =
  /(^|[\s;&|])(>{1,2}|rm\s|mv\s|cp\s|sed\s+-i|truncate\s|tee\s|dd\s)/;

export function createProtectedPathBeforeToolCallHook(
  protectedPaths: string[] = DEFAULT_PROTECTED_PATHS,
): (
  context: BeforeToolCallContext,
) => Promise<BeforeToolCallResult | undefined> {
  return async (context) => {
    if (context.toolCall.name !== "shell") {
      return undefined;
    }
    const args = context.args;
    const command =
      args && typeof args === "object" && "command" in args
        ? (args as { command: unknown }).command
        : undefined;
    if (
      typeof command !== "string" ||
      !WRITE_LOOKING_COMMAND_PATTERN.test(command)
    ) {
      return undefined;
    }
    const hit = protectedPaths.find((path) => command.includes(path));
    if (!hit) {
      return undefined;
    }
    return {
      block: true,
      reason: `PROTECTED-PATHS: command touches "${hit}" (protected: ${protectedPaths.join(", ")}). Pick another target.`,
    };
  };
}

// ---------------------------------------------------------------------------
// Seed conversion: Exo Message[] -> { systemPrompt, messages: PiMessage[] }
// Used once, up front, to seed a fresh Agent's initialState before
// Agent.continue() takes over driving the transcript itself.
// ---------------------------------------------------------------------------

export interface ExoConversationSeed {
  systemPrompt: string;
  messages: PiMessage[];
}

export function exoMessagesToAgentSeed(
  messages: Message[],
): ExoConversationSeed {
  const systemParts: string[] = [];
  const piMessages: PiMessage[] = [];
  for (const message of messages) {
    const text = messageText(message);
    if (!text) {
      continue;
    }
    if (message.role === "system" || message.role === "developer") {
      systemParts.push(text);
      continue;
    }
    if (message.role === "user") {
      piMessages.push({ role: "user", content: text, timestamp: Date.now() });
      continue;
    }
    if (message.role === "assistant") {
      piMessages.push(placeholderAssistantMessage(text));
      continue;
    }
    // role === "tool": a seeded historical tool result has no toolCallId
    // pi's Agent recognizes from its own transcript, so it can't be a real
    // ToolResultMessage. Folded into a synthetic user note instead - keeps
    // the context, isn't a faithful replay.
    piMessages.push({
      role: "user",
      content: `[prior tool result]\n${text}`,
      timestamp: Date.now(),
    });
  }
  return { systemPrompt: systemParts.join("\n\n"), messages: piMessages };
}

function placeholderAssistantMessage(text: string): AssistantMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "openai-responses",
    provider: "exo",
    model: "unknown",
    usage: EMPTY_USAGE,
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Model stub: pi's Agent wants a Model<Api> to pass to streamFn and to stamp
// onto AssistantMessage.api/provider/model. Exo's own model routing
// (secret store, provider selection) stays entirely inside
// runtimeFromModelBinding; this stub only carries the id through.
// ---------------------------------------------------------------------------

export function buildModelStub(modelId: string): PiModel<PiApi> {
  return {
    id: modelId,
    name: modelId,
    api: "openai-responses",
    provider: "exo",
    baseUrl: "",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 0,
    maxTokens: 0,
  };
}

// ---------------------------------------------------------------------------
// StreamFn adapter: wraps Exo's existing ResponsesRuntimeLike.completeStream()
// so pi's Agent drives the tool loop while Exo's model/secret/cost plumbing
// (runtimeFromModelBinding, Braintrust tracing, price table) stays untouched.
// Real token deltas: onTextDelta below forwards every chunk both into the
// pi-ai event protocol (text_start/text_delta/text_end, so anything
// consuming the Agent's own event stream sees incremental text) and, via
// options.onTextDelta, out to Exo's own context.stream.text() - the same
// call the original runResponsesTurnLoop makes - so a live REPL/websocket
// client sees tokens arrive as they're generated instead of one block at
// the end of the round.
// ---------------------------------------------------------------------------

// No round has a built-in timeout anywhere in this path - a genuinely
// stuck provider connection (not just a slow one; a live 3000s run traced
// back to OpenRouter free-tier congestion, not a hang, but nothing bounds
// the wait either way) would block the turn indefinitely. completeStream()
// doesn't take an AbortSignal, so this can only stop *waiting* on the
// underlying request, not cancel it - still turns an unbounded hang into a
// bounded one, which is what matters for the turn (and the caller) to make
// progress.
const DEFAULT_STREAM_TIMEOUT_MS = 120_000;

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`model call timed out after ${ms}ms`)),
      ms,
    );
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

export interface CreateExoStreamFnOptions {
  onFirstChunk?: (ttftMs: number) => void | Promise<void>;
  onTextDelta?: (text: string) => void | Promise<void>;
  timeoutMs?: number;
}

export function createExoStreamFn(
  runtime: ResponsesRuntimeLike,
  exoModelId: string,
  options: CreateExoStreamFnOptions = {},
): StreamFn {
  const timeoutMs = options.timeoutMs ?? DEFAULT_STREAM_TIMEOUT_MS;
  return (model, context) => {
    const stream = createAssistantMessageEventStream();
    void (async () => {
      let text = "";
      let textStarted = false;
      let started = false;
      const partial = (): AssistantMessage => ({
        role: "assistant",
        content: text ? [{ type: "text", text }] : [],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: EMPTY_USAGE,
        stopReason: "pending",
        timestamp: Date.now(),
      });
      try {
        const request = piContextToNativeRequest(exoModelId, context);
        started = true;
        stream.push({ type: "start", partial: partial() });
        const response = await withTimeout(
          runtime.completeStream(request, {
            onFirstChunk: options.onFirstChunk,
            onTextDelta: async (delta) => {
              if (!textStarted) {
                textStarted = true;
                stream.push({
                  type: "text_start",
                  contentIndex: 0,
                  partial: partial(),
                });
              }
              text += delta;
              stream.push({
                type: "text_delta",
                contentIndex: 0,
                delta,
                partial: partial(),
              });
              await options.onTextDelta?.(delta);
            },
          }),
          timeoutMs,
        );
        if (textStarted) {
          stream.push({
            type: "text_end",
            contentIndex: 0,
            content: text,
            partial: partial(),
          });
        }
        const assistantMessage = responseToAssistantMessage(response, model);
        stream.push({
          type: "done",
          reason: assistantMessage.stopReason as Extract<
            AssistantMessage["stopReason"],
            "stop" | "length" | "toolUse" | "deferred"
          >,
          message: assistantMessage,
        });
        stream.end(assistantMessage);
      } catch (error) {
        // Contract: streamFn must not throw or return a rejected promise -
        // failures are encoded in the stream itself. "start" was already
        // pushed above unless piContextToNativeRequest itself threw before
        // reaching it - only push it here in that one case, so a failure
        // mid-stream doesn't produce two "start" events.
        const errorMessage = buildErrorAssistantMessage(model, error);
        if (!started) {
          stream.push({ type: "start", partial: errorMessage });
        }
        stream.push({ type: "error", reason: "error", error: errorMessage });
        stream.end(errorMessage);
      }
    })();
    return stream;
  };
}

function piContextToNativeRequest(
  model: string,
  context: PiContext,
): NativeResponsesRequest {
  const messages: Message[] = [];
  if (context.systemPrompt) {
    messages.push({ role: "developer", content: context.systemPrompt });
  }
  for (const message of context.messages) {
    messages.push(piMessageToExoMessage(message));
  }
  return {
    model,
    messages,
    tools: (context.tools ?? []).map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters as unknown as JsonValue,
    })),
  };
}

function piMessageToExoMessage(message: PiMessage): Message {
  if (message.role === "user") {
    return { role: "user", content: piContentText(message.content) };
  }
  if (message.role === "assistant") {
    // Preserve tool-call content blocks, not just text - dropping them here
    // produces a message sequence a "tool" role message can't legally follow
    // (no matching call to point back at), which surfaced as a real bug:
    // the round after a tool call would silently fail and the turn would
    // end early with no follow-up assistant message. Exo's own
    // materializePromptMessages preserves the same shape when replaying a
    // conversation's event log, so this matches what the runtime already
    // expects.
    return { role: "assistant", content: piAssistantContentParts(message) };
  }
  // role === "toolResult": use Exo's own tool-result message shape (matches
  // what materializeEventsToMessages produces from a real tool_result event)
  // instead of a synthetic user note, and carry through the original Exo
  // ToolResult from AgentToolResult.details - toolInstanceToAgentTool sets
  // details to exactly that value, and pi-agent-core's Agent copies it
  // through onto the ToolResultMessage unchanged.
  return toolResultMessage(
    message.toolCallId,
    message.toolName,
    (message.details ?? piContentText(message.content)) as JsonValue,
  );
}

function piAssistantContentParts(message: AssistantMessage): JsonValue[] {
  const parts: JsonValue[] = [];
  for (const part of message.content) {
    if (part.type === "text" && part.text) {
      parts.push({ type: "text", text: part.text });
    } else if (part.type === "toolCall") {
      parts.push({
        type: "tool_call",
        tool_call_id: part.id,
        tool_name: part.name,
        arguments: part.arguments as JsonValue,
      });
    }
  }
  return parts;
}

// Matches Exo's own contentText()/messageText() convention (harness/index.ts)
// of standing "[image]" in for an image part when flattening to plain text -
// not full multimodal fidelity (piMessageToExoMessage always flattens to
// text, so a real image never reaches the model through this path either
// way), but at least leaves a visible trace instead of silently vanishing.
// Silent loss was the actual bug found by inspection: this filtered image
// parts out with no placeholder at all, unlike every other text-flattening
// path in this file and in Exo's own harness.
function piContentText(
  content: string | (PiTextContent | PiImageContent)[],
): string {
  if (typeof content === "string") {
    return content;
  }
  return content
    .map((part) => (part.type === "text" ? part.text : "[image]"))
    .join("");
}

export function piAssistantText(message: AssistantMessage): string {
  return message.content
    .filter((part): part is PiTextContent => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function responseToAssistantMessage(
  response: OpenAIResponse,
  model: PiModel<PiApi>,
): AssistantMessage {
  const toolCalls = responseToolCalls(response);
  const content: AssistantMessage["content"] = [];
  // response.output_text is not reliably populated across runtimes -
  // ChatCompletionsRuntime (used for OpenRouter and most non-OpenAI-Responses
  // models) leaves it undefined and only fills response.output. Go through
  // the same Lingua-backed extraction the original turn loop uses instead.
  const text = responseMessages(response).map(messageText).join("");
  if (text) {
    content.push({ type: "text", text });
  }
  for (const call of toolCalls) {
    content.push({
      type: "toolCall",
      id: call.toolCallId,
      name: call.request.functionName,
      arguments: call.request.arguments,
    });
  }
  return {
    role: "assistant",
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: usageFromResponse(response),
    stopReason: toolCalls.length > 0 ? "toolUse" : "stop",
    timestamp: Date.now(),
  };
}

function usageFromResponse(
  response: OpenAIResponse,
): AssistantMessage["usage"] {
  const usage = response.usage;
  const input = usage?.input_tokens ?? 0;
  const output = usage?.output_tokens ?? 0;
  const cacheRead = usage?.input_tokens_details?.cached_tokens ?? 0;
  return {
    input,
    output,
    cacheRead,
    cacheWrite: 0,
    totalTokens: input + output,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function buildErrorAssistantMessage(
  model: PiModel<PiApi>,
  error: unknown,
): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: EMPTY_USAGE,
    stopReason: "error",
    errorMessage: error instanceof Error ? error.message : String(error),
    timestamp: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Unfinished-turn detection: benchmarking against a live free/weak model
// (see harness-pi-core.ts) found two ways a turn ends with zero tool results
// while the task plainly isn't done:
//   - the model emits its tool call as plain text (e.g. "[TOOL_CALL shell]
//     {...}" or a leaked "</minimax:tool_call>" chat-template token) instead
//     of a real structured tool call - content has no "toolCall" part, so
//     the turn looks like ordinary text output.
//   - the model returns a genuinely empty message: no text, no tool call,
//     stopReason "stop" - this is exactly the shape hit live on exo-bench's
//     own (unmodified) turn loop on the t5-pipeline case: it wrote three
//     scripts, then a later round came back empty and the turn just ended
//     with no final message. Exo's loop has no way to tell "empty" apart
//     from "a legitimately short final answer" either - this harness has
//     the same blind spot, just via a different code path.
// In both cases pi-agent-core's Agent sees zero tool results and stops -
// unlike Exo's own turn loop, which at least hits an explicit parse error
// from the model API on a malformed tool call and keeps looping until it
// gets a real one (though not on a truly empty response, which is exactly
// the exo-bench failure above). Same underlying model flakiness, different
// failure shape depending on the path. This heuristic lets the harness
// recognize "that wasn't actually a finished answer" and nudge a retry via
// agent.followUp() instead of ending the turn early. False positives (a
// genuine final answer that happens to mention "tool call", or is
// legitimately terse) are the deliberate failure mode here - one extra
// clarifying round costs far less than silently abandoning the task.
//
// Deliberately excludes stopReason "error"/"aborted": pi-agent-core's Agent
// returns immediately for those (agent-loop.js's runLoop never reaches the
// getFollowUpMessages() check on that path), so a followUp() call here
// would be queued and then never read. See the "error" branch of
// piEventToExoEvents for how that case is handled instead - recorded, not
// retried.
// ---------------------------------------------------------------------------

const MALFORMED_TOOL_CALL_TEXT_PATTERN = /tool[_ ]call/i;

// Accepts a loose shape rather than pi-ai's AssistantMessage: the caller
// receives pi-agent-core's AgentMessage, a union that also admits whatever
// custom message types other packages declaration-merge into
// CustomAgentMessages - not every member has a "content" array of the shape
// AssistantMessage promises, so this checks defensively instead of trusting
// the type.
export function looksLikeUnfinishedTurn(message: {
  role: string;
  content?: unknown;
  stopReason?: unknown;
}): boolean {
  if (
    message.role !== "assistant" ||
    !Array.isArray(message.content) ||
    message.stopReason === "error" ||
    message.stopReason === "aborted"
  ) {
    return false;
  }
  const parts = message.content as Array<Record<string, unknown>>;
  const hasRealToolCall = parts.some((part) => part?.type === "toolCall");
  if (hasRealToolCall) {
    return false;
  }
  const text = parts
    .filter(
      (part): part is { type: "text"; text: string } =>
        part?.type === "text" && typeof part.text === "string",
    )
    .map((part) => part.text)
    .join("");
  return (
    text.trim().length === 0 || MALFORMED_TOOL_CALL_TEXT_PATTERN.test(text)
  );
}

// ---------------------------------------------------------------------------
// Event translation: pi AgentEvent -> Exo EventData[]. Mirrors the shape
// exoharness/examples/typescript/pi-harness.ts already uses for the
// subprocess integration (eventsForPiEvent) - same event names, because
// pi-coding-agent's --mode json output is this same AgentEvent protocol
// serialized to JSON lines. Only message_end and the tool_execution_*
// pair carry anything Exo's event log needs to durably record.
// ---------------------------------------------------------------------------

export interface PiToolExecutionStartEvent {
  type: "tool_execution_start";
  toolCallId: string;
  toolName: string;
  args: unknown;
}

export interface PiToolExecutionEndEvent {
  type: "tool_execution_end";
  toolCallId: string;
  toolName: string;
  result: unknown;
  isError: boolean;
}

export interface PiMessageEndEvent {
  type: "message_end";
  message: PiMessage;
}

export type PiRecordableEvent =
  | PiToolExecutionStartEvent
  | PiToolExecutionEndEvent
  | PiMessageEndEvent;

export function piEventToExoEvents(event: PiRecordableEvent): EventData[] {
  if (event.type === "tool_execution_start") {
    return [
      toolRequestedEvent({
        toolCallId: event.toolCallId,
        request: {
          functionName: event.toolName,
          arguments: (event.args ?? {}) as JsonObject,
        },
      }),
    ];
  }
  if (event.type === "tool_execution_end") {
    return [
      toolResultEvent(event.toolCallId, {
        ok: !event.isError,
        result: (event.result ?? null) as JsonValue,
      }),
    ];
  }
  if (event.message.role !== "assistant") {
    return [];
  }
  // A round that failed at the streamFn level (see createExoStreamFn's
  // catch block) surfaces here as stopReason "error" with empty content.
  // pi-agent-core's Agent returns immediately in this case - it never
  // reaches the follow-up check, so looksLikeUnfinishedTurn's retry nudge
  // can't help (see its comment). Record the failure instead of silently
  // dropping it: an operator reading the event log should see why a turn
  // ended with no answer, not nothing at all.
  if (event.message.stopReason === "error") {
    return [
      messagesEvent([
        assistantTextMessage(
          `[turn ended: model call failed] ${event.message.errorMessage ?? "unknown error"}`,
        ),
      ]),
    ];
  }
  const text = piAssistantText(event.message);
  if (!text) {
    // A pure tool-call message (no text) is already recorded via
    // tool_execution_start/end - nothing extra needed. But a genuinely
    // empty, non-error assistant turn (no text, no tool call, stopReason
    // "stop") would otherwise vanish with zero trace, even after every
    // looksLikeUnfinishedTurn retry nudge also comes back empty - record it
    // so an operator reading the event log sees why the turn ended blank.
    const hasToolCall = event.message.content.some(
      (part) => part.type === "toolCall",
    );
    if (hasToolCall) {
      return [];
    }
    return [
      messagesEvent([
        assistantTextMessage("[turn ended: empty response from model]"),
      ]),
    ];
  }
  return [
    messagesEvent(
      [assistantTextMessage(text)],
      undefined,
      usageToJsonObject(event.message.usage, event.message.model),
    ),
  ];
}

function usageToJsonObject(
  usage: AssistantMessage["usage"],
  model: string,
): JsonObject {
  return {
    model,
    prompt_tokens: usage.input,
    completion_tokens: usage.output,
    prompt_cached_tokens: usage.cacheRead,
  };
}
