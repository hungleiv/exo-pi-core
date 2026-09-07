import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ToolInstance, TurnContext } from "@exo/harness";
import type { ResponsesRuntimeLike } from "@exo/model-runtime/responses";
import { describe, expect, it } from "vitest";

import {
  buildModelStub,
  createExoStreamFn,
  createProtectedPathBeforeToolCallHook,
  exoMessagesToAgentSeed,
  looksLikeUnfinishedTurn,
  piEventToExoEvents,
  toolInstanceToAgentTool,
} from "./pi-agent-adapters";

function fakeContext(): TurnContext {
  return {} as TurnContext;
}

describe("toolInstanceToAgentTool", () => {
  function tool(execute: ToolInstance["handler"]["execute"]): ToolInstance {
    return {
      definition: {
        name: "echo",
        description: "Echoes input.",
        parameters: {
          type: "object",
          additionalProperties: false,
          properties: { text: { type: "string" } },
          required: ["text"],
        },
      },
      source: "library",
      handler: { execute },
    };
  }

  it("carries name/description/parameters through", () => {
    const agentTool = toolInstanceToAgentTool(
      tool(async () => ({ ok: true, value: "x" })),
      fakeContext(),
    );
    expect(agentTool.name).toBe("echo");
    expect(agentTool.label).toBe("echo");
    expect(agentTool.description).toBe("Echoes input.");
    expect(agentTool.parameters).toEqual({
      type: "object",
      additionalProperties: false,
      properties: { text: { type: "string" } },
      required: ["text"],
    });
  });

  it("wraps a successful Exo result into AgentToolResult content+details", async () => {
    const agentTool = toolInstanceToAgentTool(
      tool(async (args) => ({ ok: true, echoed: args.text })),
      fakeContext(),
    );
    const result = await agentTool.execute("call-1", { text: "hi" });
    expect(result.details).toEqual({ ok: true, echoed: "hi" });
    expect(result.content).toEqual([
      {
        type: "text",
        text: JSON.stringify({ ok: true, echoed: "hi" }, null, 2),
      },
    ]);
  });

  it("throws when the Exo handler reports ok: false (pi's failure convention)", async () => {
    const agentTool = toolInstanceToAgentTool(
      tool(async () => ({ ok: false, error: "boom" })),
      fakeContext(),
    );
    await expect(agentTool.execute("call-1", {})).rejects.toThrow("boom");
  });

  it("passes toolCallId and the turn context through to the handler", async () => {
    const context = fakeContext();
    let seenCallId: string | undefined;
    let seenContext: unknown;
    const agentTool = toolInstanceToAgentTool(
      tool(async (_args, execution) => {
        seenCallId = execution.toolCallId;
        seenContext = execution.context;
        return { ok: true };
      }),
      context,
    );
    await agentTool.execute("call-42", {});
    expect(seenCallId).toBe("call-42");
    expect(seenContext).toBe(context);
  });

  it("passes a small result through unchanged, with no artifact written", async () => {
    let writeArtifactTextCalls = 0;
    const context = {
      exoharness: {
        current: {
          turn: {
            writeArtifactText: async () => {
              writeArtifactTextCalls += 1;
              throw new Error("should not be called for a small result");
            },
          },
        },
      },
    } as unknown as TurnContext;
    const agentTool = toolInstanceToAgentTool(
      tool(async () => ({ ok: true, value: "small" })),
      context,
    );
    const result = await agentTool.execute("call-1", {});
    expect(result.details).toEqual({ ok: true, value: "small" });
    expect(writeArtifactTextCalls).toBe(0);
  });

  it("truncates a large result to an artifact instead of putting it all in context", async () => {
    const bigValue = "x".repeat(20_000);
    let writtenPath: string | undefined;
    let writtenText: string | undefined;
    const context = {
      exoharness: {
        current: {
          turn: {
            writeArtifactText: async (args: { path: string; text: string }) => {
              writtenPath = args.path;
              writtenText = args.text;
              return {
                artifactId: "artifact-1",
                path: args.path,
                version: 1,
                createdAt: "now",
                sizeBytes: args.text.length,
              };
            },
          },
        },
      },
    } as unknown as TurnContext;
    const agentTool = toolInstanceToAgentTool(
      tool(async () => ({ ok: true, value: bigValue })),
      context,
    );
    const result = await agentTool.execute("call-1", {});

    expect(writtenPath).toBe("tool-results/echo/call-1/result.json");
    expect(writtenText).toContain(bigValue);

    const text = result.content[0];
    expect(text?.type).toBe("text");
    expect(text && "text" in text ? text.text.length : 0).toBeLessThan(
      writtenText?.length ?? Infinity,
    );
    expect(text && "text" in text ? text.text : "").toContain(
      "full result written to artifact artifact-1",
    );
    expect(result.details).toMatchObject({
      truncated: true,
      artifactId: "artifact-1",
    });
  });
  // The failure class that cost ~$9: a model copies a wrapper shape it saw in
  // its own history, or double-encodes the arguments object as a string, and
  // every call then fails schema validation. prepareArguments runs before
  // validation, so a salvageable call still executes instead of burning a
  // round on "missing field".
  it("repairs double-encoded and wrapper-nested tool arguments", async () => {
    const seen: Record<string, unknown>[] = [];
    const agentTool = toolInstanceToAgentTool(
      tool(async (args) => {
        seen.push(args);
        return { ok: true };
      }),
      fakeContext(),
    );

    const cases: unknown[] = [
      { text: "plain" },
      JSON.stringify({ text: "encoded-once" }),
      { type: "valid", value: { text: "wrapped" } },
      JSON.stringify({ type: "valid", value: { text: "wrapped-and-encoded" } }),
      { type: "valid", value: { type: "valid", value: { text: "twice" } } },
    ];
    for (const raw of cases) {
      const prepared = agentTool.prepareArguments?.(raw) ?? raw;
      await agentTool.execute("call_1", prepared as never);
    }

    expect(seen).toEqual([
      { text: "plain" },
      { text: "encoded-once" },
      { text: "wrapped" },
      { text: "wrapped-and-encoded" },
      { text: "twice" },
    ]);
  });

  // Repair must not paper over a genuinely wrong shape - that would hide a
  // real schema mismatch instead of surfacing it.
  it("leaves unrecognised argument shapes untouched", () => {
    const agentTool = toolInstanceToAgentTool(
      tool(async (args) => args),
      fakeContext(),
    );

    expect(agentTool.prepareArguments?.("not json")).toBe("not json");
    expect(agentTool.prepareArguments?.({ text: "fine" })).toEqual({
      text: "fine",
    });
    // A two-key object that merely looks like the wrapper is not one.
    expect(agentTool.prepareArguments?.({ type: "other", value: 1 })).toEqual({
      type: "other",
      value: 1,
    });
  });
});

describe("exoMessagesToAgentSeed", () => {
  it("folds system and developer messages into systemPrompt, in order", () => {
    const seed = exoMessagesToAgentSeed([
      { role: "system", content: "base rules" },
      { role: "developer", content: "harness notes" },
      { role: "user", content: "hello" },
    ]);
    expect(seed.systemPrompt).toBe("base rules\n\nharness notes");
    expect(seed.messages).toEqual([
      { role: "user", content: "hello", timestamp: expect.any(Number) },
    ]);
  });

  it("converts a user/assistant pair into pi-ai Message objects", () => {
    const seed = exoMessagesToAgentSeed([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello there" },
    ]);
    expect(seed.messages).toHaveLength(2);
    expect(seed.messages[0]).toMatchObject({ role: "user", content: "hi" });
    const assistant = seed.messages[1] as AssistantMessage;
    expect(assistant.role).toBe("assistant");
    expect(assistant.content).toEqual([{ type: "text", text: "hello there" }]);
    expect(assistant.stopReason).toBe("stop");
  });

  it("folds a prior tool-result message into a synthetic user note", () => {
    const seed = exoMessagesToAgentSeed([
      { role: "tool", content: "42", id: "call-1" },
    ]);
    expect(seed.messages).toEqual([
      {
        role: "user",
        content: "[prior tool result]\n42",
        timestamp: expect.any(Number),
      },
    ]);
  });

  it("drops messages with no extractable text", () => {
    const seed = exoMessagesToAgentSeed([{ role: "user", content: "" }]);
    expect(seed.messages).toEqual([]);
  });
  // Same leak as responseToAssistantMessage, on the across-turns path: a prior
  // assistant turn's tool calls must not be seeded back as "[tool_call ...]"
  // prose for the model to imitate. The tool result note still carries what
  // the turn actually did.
  it("does not seed prior tool calls as prose", () => {
    const seed = exoMessagesToAgentSeed([
      { role: "user", content: "run it" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "on it" },
          {
            type: "tool_call",
            tool_call_id: "call_1",
            tool_name: "shell",
            arguments: { command: "echo hi" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool_result",
            tool_call_id: "call_1",
            tool_name: "shell",
            output: { exit_code: 0, stdout: "hi\n", stderr: "" },
          },
        ],
      },
    ] as never);

    const assistant = seed.messages.filter((m) => m.role === "assistant");
    expect(assistant).toHaveLength(1);
    expect(JSON.stringify(assistant[0].content)).not.toContain("[tool_call");
    expect(JSON.stringify(assistant[0].content)).toContain("on it");
    // The tool result still reaches the model.
    expect(
      seed.messages.some((m) =>
        String(m.content).includes("[prior tool result]"),
      ),
    ).toBe(true);
  });
});

describe("buildModelStub", () => {
  it("carries the model id through with safe defaults", () => {
    const model = buildModelStub("gpt-5.5");
    expect(model.id).toBe("gpt-5.5");
    expect(model.provider).toBe("exo");
    expect(model.contextWindow).toBe(0);
  });
});

describe("piEventToExoEvents", () => {
  it("translates tool_execution_start into a tool_requested event", () => {
    const events = piEventToExoEvents({
      type: "tool_execution_start",
      toolCallId: "call-1",
      toolName: "echo",
      args: { text: "hi" },
    });
    expect(events).toEqual([
      {
        type: "tool_requested",
        tool_call_id: "call-1",
        response_id: undefined,
        request: { function_name: "echo", arguments: { text: "hi" } },
      },
    ]);
  });

  it("translates tool_execution_end into a tool_result event carrying ok/isError", () => {
    const events = piEventToExoEvents({
      type: "tool_execution_end",
      toolCallId: "call-1",
      toolName: "echo",
      result: { echoed: "hi" },
      isError: false,
    });
    expect(events).toEqual([
      {
        type: "tool_result",
        tool_call_id: "call-1",
        result: { ok: true, result: { echoed: "hi" } },
      },
    ]);
  });

  it("marks a failed tool_execution_end as not ok", () => {
    const events = piEventToExoEvents({
      type: "tool_execution_end",
      toolCallId: "call-1",
      toolName: "echo",
      result: "boom",
      isError: true,
    });
    expect(events[0]).toMatchObject({
      result: { ok: false, result: "boom" },
    });
  });

  it("translates an assistant message_end into a messages event with usage", () => {
    const events = piEventToExoEvents({
      type: "message_end",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "hello" }],
        api: "openai-responses",
        provider: "exo",
        model: "gpt-5.5",
        usage: {
          input: 10,
          output: 5,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 15,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: Date.now(),
      },
    });
    expect(events).toEqual([
      {
        type: "messages",
        messages: [{ role: "assistant", content: "hello" }],
        response_id: undefined,
        usage: {
          model: "gpt-5.5",
          prompt_tokens: 10,
          completion_tokens: 5,
          prompt_cached_tokens: 0,
        },
      },
    ]);
  });

  it("ignores a message_end for a non-assistant message", () => {
    const events = piEventToExoEvents({
      type: "message_end",
      message: { role: "user", content: "hi", timestamp: Date.now() },
    });
    expect(events).toEqual([]);
  });

  it("records a genuinely empty assistant message_end instead of dropping it", () => {
    // No text, no tool call, stopReason "stop" - the shape that vanished
    // with zero trace before this was fixed (pi-core-bench t3-fizzbuzz on
    // 2026-09-04: 0 rounds, empty final text, nothing in the event log to
    // explain why).
    const events = piEventToExoEvents({
      type: "message_end",
      message: {
        role: "assistant",
        content: [],
        api: "openai-responses",
        provider: "exo",
        model: "gpt-5.5",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: Date.now(),
      },
    });
    expect(events).toEqual([
      {
        type: "messages",
        messages: [
          {
            role: "assistant",
            content: "[turn ended: empty response from model]",
          },
        ],
        response_id: undefined,
      },
    ]);
  });

  it("ignores a pure tool-call message_end with no text (already recorded via tool_execution_start/end)", () => {
    const events = piEventToExoEvents({
      type: "message_end",
      message: {
        role: "assistant",
        content: [
          { type: "toolCall", id: "call-1", name: "shell", arguments: {} },
        ],
        api: "openai-responses",
        provider: "exo",
        model: "gpt-5.5",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "toolUse",
        timestamp: Date.now(),
      },
    });
    expect(events).toEqual([]);
  });

  it("records a stopReason error as a visible diagnostic message instead of dropping it", () => {
    const events = piEventToExoEvents({
      type: "message_end",
      message: {
        role: "assistant",
        content: [],
        api: "openai-responses",
        provider: "exo",
        model: "gpt-5.5",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "error",
        errorMessage: "provider unavailable",
        timestamp: Date.now(),
      },
    });
    expect(events).toEqual([
      {
        type: "messages",
        messages: [
          {
            role: "assistant",
            content: "[turn ended: model call failed] provider unavailable",
          },
        ],
        response_id: undefined,
      },
    ]);
  });
});

describe("createExoStreamFn", () => {
  const model = buildModelStub("gpt-5.5");

  function fakeResponse(text: string) {
    return {
      // output_text is deliberately omitted: ChatCompletionsRuntime
      // (OpenRouter and most non-OpenAI-Responses models) never populates
      // it - only response.output does, in this shape.
      output: [
        {
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text, annotations: [] }],
        },
      ],
      usage: { input_tokens: 3, output_tokens: 2 },
      // minimal fake Response, only the fields the adapter reads are
      // populated.
    } as never;
  }

  it("stands in [image] for an image part instead of silently dropping it", async () => {
    let seenUserContent: unknown;
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream(request) {
        seenUserContent = request.messages?.find(
          (m) => m.role === "user",
        )?.content;
        return fakeResponse("ok");
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
    );
    const stream = await streamFn(model, {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look: " },
            { type: "image", data: "base64...", mimeType: "image/png" },
          ],
          timestamp: Date.now(),
        },
      ],
    });
    for await (const _event of stream) {
      // drain
    }
    expect(seenUserContent).toBe("look: [image]");
  });

  // Regression: messageText() renders a tool_call content part as the literal
  // string "[tool_call <name>] {<args>}". Running the model's own response
  // through it put every tool call into the transcript twice - once as that
  // fake prose, once as the real structured call - and the model imitated the
  // prose form, producing turns that called nothing and burned the
  // unfinished-turn nudge budget. The assistant text must carry only what the
  // model actually wrote.
  it("keeps tool calls out of the assistant text", async () => {
    const responseWithToolCall = {
      output: [
        {
          type: "message",
          role: "assistant",
          status: "completed",
          content: [
            { type: "output_text", text: "running it", annotations: [] },
          ],
        },
        {
          id: "call_1_item",
          type: "function_call",
          call_id: "call_1",
          name: "shell",
          arguments: JSON.stringify({ command: "echo hi" }),
          status: "completed",
        },
      ],
      usage: { input_tokens: 3, output_tokens: 2 },
    } as never;
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream() {
        return responseWithToolCall;
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
    );
    const stream = await streamFn(model, {
      messages: [{ role: "user", content: "go", timestamp: Date.now() }],
    });
    let final: AssistantMessage | undefined;
    for await (const event of stream) {
      if (event.type === "done") {
        final = event.message;
      }
    }

    const textParts = (final?.content ?? []).flatMap((part) =>
      part.type === "text" ? [part.text] : [],
    );
    expect(textParts.join("")).toBe("running it");
    expect(textParts.join("")).not.toContain("[tool_call");
    // The real structured call still has to survive.
    expect(
      (final?.content ?? []).filter((part) => part.type === "toolCall"),
    ).toHaveLength(1);
  });

  it("streams real deltas as text_start/text_delta/text_end, then start+done", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream(_request, handlers) {
        await handlers?.onFirstChunk?.(42);
        await handlers?.onTextDelta?.("hi ");
        await handlers?.onTextDelta?.("there");
        return fakeResponse("hi there");
      },
    };
    const seenDeltas: string[] = [];
    let ttft: number | undefined;
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
      {
        onFirstChunk: (ms) => {
          ttft = ms;
        },
        onTextDelta: (text) => {
          seenDeltas.push(text);
        },
      },
    );
    const stream = await streamFn(model, { messages: [] });

    const events = [];
    for await (const event of stream) {
      events.push(event);
    }
    expect(events.map((event) => event.type)).toEqual([
      "start",
      "text_start",
      "text_delta",
      "text_delta",
      "text_end",
      "done",
    ]);
    expect(ttft).toBe(42);
    // options.onTextDelta (-> context.stream.text in the real harness) sees
    // every raw delta, not just the final accumulated text.
    expect(seenDeltas).toEqual(["hi ", "there"]);

    const done = events[5] as { type: "done"; message: AssistantMessage };
    expect(done.message.content).toEqual([{ type: "text", text: "hi there" }]);
    expect(done.message.stopReason).toBe("stop");
    expect(done.message.usage.input).toBe(3);
    expect(done.message.usage.output).toBe(2);

    const result = await stream.result();
    expect(result.content).toEqual([{ type: "text", text: "hi there" }]);
  });

  it("skips text_start/text_end when the response has no text deltas at all", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream() {
        return fakeResponse("");
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
    );
    const stream = await streamFn(model, { messages: [] });

    const events = [];
    for await (const event of stream) {
      events.push(event);
    }
    expect(events.map((event) => event.type)).toEqual(["start", "done"]);
  });

  it("never throws: encodes a runtime failure as a start+error pair", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream() {
        throw new Error("provider unavailable");
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
    );
    const stream = await streamFn(model, { messages: [] });

    const events = [];
    for await (const event of stream) {
      events.push(event);
    }
    expect(events.map((event) => event.type)).toEqual(["start", "error"]);
    const errored = events[1] as { type: "error"; error: AssistantMessage };
    expect(errored.error.stopReason).toBe("error");
    expect(errored.error.errorMessage).toBe("provider unavailable");
  });

  it("bounds a stuck provider call instead of waiting forever", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      completeStream() {
        return new Promise(() => {
          // never resolves - simulates a genuinely stuck connection.
        });
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
      {
        timeoutMs: 20,
      },
    );
    const stream = await streamFn(model, { messages: [] });

    const events = [];
    for await (const event of stream) {
      events.push(event);
    }
    expect(events.map((event) => event.type)).toEqual(["start", "error"]);
    const errored = events[1] as { type: "error"; error: AssistantMessage };
    expect(errored.error.errorMessage).toBe("model call timed out after 20ms");
  });
});

describe("createProtectedPathBeforeToolCallHook", () => {
  const hook = createProtectedPathBeforeToolCallHook();

  function toolCallContext(name: string, args: unknown) {
    return { toolCall: { name }, args } as Parameters<typeof hook>[0];
  }

  it("blocks a redirect into a protected path", async () => {
    const result = await hook(
      toolCallContext("shell", { command: "echo secret > .env" }),
    );
    expect(result?.block).toBe(true);
    expect(result?.reason).toContain(".env");
  });

  it("blocks rm targeting a protected path", async () => {
    const result = await hook(
      toolCallContext("shell", { command: "rm -rf .git/hooks" }),
    );
    expect(result?.block).toBe(true);
  });

  it("allows a plain read of a protected path", async () => {
    const result = await hook(
      toolCallContext("shell", { command: "cat .env" }),
    );
    expect(result).toBeUndefined();
  });

  it("allows writes outside any protected path", async () => {
    const result = await hook(
      toolCallContext("shell", { command: "echo hi > notes.txt" }),
    );
    expect(result).toBeUndefined();
  });

  it("ignores tool calls that aren't shell", async () => {
    const result = await hook(
      toolCallContext("web_fetch", { url: "https://example.com" }),
    );
    expect(result).toBeUndefined();
  });

  it("honors a custom protected-path list", async () => {
    const customHook = createProtectedPathBeforeToolCallHook(["secrets/"]);
    const blocked = await customHook(
      toolCallContext("shell", { command: "rm secrets/prod.key" }),
    );
    expect(blocked?.block).toBe(true);
    const allowed = await customHook(
      toolCallContext("shell", { command: "rm .env" }),
    );
    expect(allowed).toBeUndefined();
  });
});

describe("looksLikeUnfinishedTurn", () => {
  it("flags an assistant message with no real tool call but tool-call-shaped text", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [{ type: "text", text: '[TOOL_CALL shell] {"command":"ls"}' }],
      }),
    ).toBe(true);
  });

  it("flags a leaked chat-template tool-call token", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [{ type: "text", text: "done</minimax:tool_call>" }],
      }),
    ).toBe(true);
  });

  it("flags a genuinely empty response (the exo-bench t5-pipeline failure shape)", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [],
        stopReason: "stop",
      }),
    ).toBe(true);
  });

  it("does not flag an empty message with stopReason error - followUp() can't help there", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [],
        stopReason: "error",
      }),
    ).toBe(false);
  });

  it("does not flag an empty message with stopReason aborted", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [],
        stopReason: "aborted",
      }),
    ).toBe(false);
  });

  it("does not flag a message that has a real toolCall content part", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [
          { type: "text", text: "I'll call the tool now." },
          { type: "toolCall", id: "1", name: "shell", arguments: {} },
        ],
      }),
    ).toBe(false);
  });

  it("does not flag a genuine plain-text final answer", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "assistant",
        content: [{ type: "text", text: "The answer is 42." }],
      }),
    ).toBe(false);
  });

  it("does not flag non-assistant messages", () => {
    expect(
      looksLikeUnfinishedTurn({
        role: "user",
        content: "tool_call whatever",
      }),
    ).toBe(false);
  });

  it("does not flag a message with no content array (e.g. a custom AgentMessage variant)", () => {
    expect(looksLikeUnfinishedTurn({ role: "assistant" })).toBe(false);
  });
});

describe("image tool results", () => {
  const model = buildModelStub("gpt-5.5");

  function fakeResponse() {
    return {
      output: [
        {
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "ok", annotations: [] }],
        },
      ],
      usage: { input_tokens: 1, output_tokens: 1 },
    } as never;
  }

  function imageToolResult(id: string, data: string) {
    return {
      role: "toolResult",
      toolCallId: id,
      toolName: "read",
      content: [
        { type: "text", text: "Read image file /tmp/a.png [image/png]" },
        { type: "image", data, mimeType: "image/png" },
      ],
      details: { path: "/tmp/a.png", media_type: "image/png", bytes: 3 },
      timestamp: Date.now(),
    };
  }

  async function capture(messages: unknown[]) {
    let seen: unknown;
    const runtime: Pick<ResponsesRuntimeLike, "completeStream"> = {
      async completeStream(request) {
        seen = request.messages;
        return fakeResponse();
      },
    };
    const streamFn = createExoStreamFn(
      runtime as ResponsesRuntimeLike,
      "gpt-5.5",
    );
    const stream = await streamFn(model, { messages } as never);
    for await (const _event of stream) {
      // drain
    }
    return seen as { role: string; content: unknown }[];
  }

  // A "tool" role message's content must be a string on the OpenAI chat API,
  // which is the path OpenRouter takes, so the image cannot ride inside the
  // tool result the way it does on Anthropic's API. It follows as a user
  // message instead - the one role whose content may be typed parts.
  it("sends the image as a user message after the tool result", async () => {
    const messages = await capture([imageToolResult("c1", "QUJD")]);

    expect(messages.map((m) => m.role)).toEqual(["tool", "user"]);
    expect(messages[1].content).toEqual([
      { type: "text", text: "Image content of the read result above:" },
      { type: "image", image: "QUJD", media_type: "image/png" },
    ]);
  });

  it("keeps the base64 out of the tool message itself", async () => {
    const messages = await capture([imageToolResult("c1", "QUJD")]);

    expect(JSON.stringify(messages[0])).not.toContain("QUJD");
  });

  // A tool result stays in pi's context for the rest of the turn and the
  // request is rebuilt every round, so without this an image is re-uploaded
  // on every round until the turn ends.
  it("sends only the newest image when several have been read", async () => {
    const messages = await capture([
      imageToolResult("c1", "T0xE"),
      imageToolResult("c2", "TkVX"),
    ]);

    const images = JSON.stringify(messages);
    expect(images).toContain("TkVX");
    expect(images).not.toContain("T0xE");
    // The older result keeps its text line, so the model still knows it read
    // that file - it just cannot re-examine the picture without reading again.
    expect(messages.map((m) => m.role)).toEqual(["tool", "tool", "user"]);
  });

  it("leaves a text-only tool result as a single message", async () => {
    const messages = await capture([
      {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "shell",
        content: [{ type: "text", text: "hi" }],
        details: { stdout: "hi" },
        timestamp: Date.now(),
      },
    ]);

    expect(messages.map((m) => m.role)).toEqual(["tool"]);
  });
});
