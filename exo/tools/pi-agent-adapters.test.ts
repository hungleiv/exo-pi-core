import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ToolInstance, TurnContext } from "@exo/harness";
import type { ResponsesRuntimeLike } from "@exo/model-runtime/responses";
import { describe, expect, it } from "vitest";

import {
  buildModelStub,
  createExoStreamFn,
  exoMessagesToAgentSeed,
  looksLikeMalformedToolCallAttempt,
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

  it("ignores an assistant message_end with no text content", () => {
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
    expect(events).toEqual([]);
  });
});

describe("createExoStreamFn", () => {
  const model = buildModelStub("gpt-5.5");

  it("replays a successful runtime.complete() as a start+done pair", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "complete"> = {
      async complete() {
        return {
          // output_text is deliberately omitted here: ChatCompletionsRuntime
          // (OpenRouter and most non-OpenAI-Responses models) never
          // populates it - only response.output does, in this shape.
          output: [
            {
              type: "message",
              role: "assistant",
              status: "completed",
              content: [
                { type: "output_text", text: "hi there", annotations: [] },
              ],
            },
          ],
          usage: {
            input_tokens: 3,
            output_tokens: 2,
          },
          // minimal fake Response, only the fields the adapter reads are
          // populated.
        } as never;
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
    const done = events[1] as { type: "done"; message: AssistantMessage };
    expect(done.message.content).toEqual([{ type: "text", text: "hi there" }]);
    expect(done.message.stopReason).toBe("stop");
    expect(done.message.usage.input).toBe(3);
    expect(done.message.usage.output).toBe(2);

    const result = await stream.result();
    expect(result.content).toEqual([{ type: "text", text: "hi there" }]);
  });

  it("never throws: encodes a runtime failure as a start+error pair", async () => {
    const runtime: Pick<ResponsesRuntimeLike, "complete"> = {
      async complete() {
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
});

describe("looksLikeMalformedToolCallAttempt", () => {
  it("flags an assistant message with no real tool call but tool-call-shaped text", () => {
    expect(
      looksLikeMalformedToolCallAttempt({
        role: "assistant",
        content: [{ type: "text", text: '[TOOL_CALL shell] {"command":"ls"}' }],
      }),
    ).toBe(true);
  });

  it("flags a leaked chat-template tool-call token", () => {
    expect(
      looksLikeMalformedToolCallAttempt({
        role: "assistant",
        content: [{ type: "text", text: "done</minimax:tool_call>" }],
      }),
    ).toBe(true);
  });

  it("does not flag a message that has a real toolCall content part", () => {
    expect(
      looksLikeMalformedToolCallAttempt({
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
      looksLikeMalformedToolCallAttempt({
        role: "assistant",
        content: [{ type: "text", text: "The answer is 42." }],
      }),
    ).toBe(false);
  });

  it("does not flag non-assistant messages", () => {
    expect(
      looksLikeMalformedToolCallAttempt({
        role: "user",
        content: "tool_call whatever",
      }),
    ).toBe(false);
  });

  it("does not flag a message with no content array (e.g. a custom AgentMessage variant)", () => {
    expect(looksLikeMalformedToolCallAttempt({ role: "assistant" })).toBe(
      false,
    );
  });
});
