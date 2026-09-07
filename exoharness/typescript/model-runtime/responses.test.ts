import { describe, expect, it } from "vitest";
import type { Response } from "openai/resources/responses/responses";

import {
  AnthropicRuntime,
  ChatCompletionsRuntime,
  linguaMessagesToResponsesInput,
  messagesToChatMessages,
  isAnthropicModel,
  isOpenRouterBinding,
  modelRequiresResponsesApi,
  responseToLinguaEvents,
  responseToolCalls,
  runtimeFromModelBinding,
  ResponsesRuntime,
} from "./responses";

describe("model runtime dispatch", () => {
  it("matches the Responses-required model families", () => {
    for (const model of [
      "o1-pro",
      "o3-pro",
      "gpt-5-pro",
      "gpt-5.3",
      "gpt-5.4",
      "gpt-5-codex",
      "gpt-5.1-codex-mini",
    ]) {
      expect(modelRequiresResponsesApi(model)).toBe(true);
    }

    for (const model of [
      "deepseek-chat",
      "gpt-4o",
      "gpt-5",
      "gpt-5.1",
      "gpt-5.2-chat-latest",
    ]) {
      expect(modelRequiresResponsesApi(model)).toBe(false);
    }
  });

  it("dispatches chat-only models away from Responses", () => {
    expect(
      runtimeFromModelBinding(undefined, {
        model: "deepseek-chat",
        apiKey: "key",
      }),
    ).toBeInstanceOf(ChatCompletionsRuntime);
    expect(
      runtimeFromModelBinding(undefined, {
        model: "gpt-5.4",
        apiKey: "key",
      }),
    ).toBeInstanceOf(ResponsesRuntime);
  });

  it("dispatches claude models to the native Anthropic runtime", () => {
    expect(isAnthropicModel("claude-sonnet-4-6")).toBe(true);
    expect(isAnthropicModel("gpt-5.4")).toBe(false);
    expect(isAnthropicModel("us.anthropic.claude-sonnet-4-6")).toBe(false);
    expect(
      runtimeFromModelBinding(undefined, {
        model: "claude-sonnet-4-6",
        apiKey: "key",
      }),
    ).toBeInstanceOf(AnthropicRuntime);
  });

  it("routes OpenRouter bindings through chat completions by base URL", () => {
    expect(
      isOpenRouterBinding({ baseUrl: "https://openrouter.ai/api/v1" }),
    ).toBe(true);
    expect(isOpenRouterBinding({ baseUrl: null })).toBe(false);
    // A Responses-looking model name over OpenRouter still uses chat completions.
    expect(
      runtimeFromModelBinding(undefined, {
        model: "openai/gpt-5-pro",
        apiKey: "key",
        baseUrl: "https://openrouter.ai/api/v1",
      }),
    ).toBeInstanceOf(ChatCompletionsRuntime);
  });
});

describe("response tool-call parsing", () => {
  it("attaches response usage to message events", () => {
    const response = {
      id: "resp_1",
      model: "gpt-5.4",
      output: [
        {
          type: "message",
          role: "assistant",
          content: [
            {
              type: "output_text",
              text: "hello",
              annotations: [],
            },
          ],
        },
      ],
      usage: {
        input_tokens: 12,
        output_tokens: 5,
        total_tokens: 17,
        input_tokens_details: {
          cached_tokens: 3,
        },
        output_tokens_details: {
          reasoning_tokens: 2,
        },
      },
    } as unknown as Response;

    expect(responseToLinguaEvents(response)).toContainEqual({
      type: "messages",
      messages: expect.any(Array),
      response_id: undefined,
      usage: expect.objectContaining({
        model: "gpt-5.4",
        prompt_tokens: 12,
        completion_tokens: 5,
        prompt_cached_tokens: 3,
        completion_reasoning_tokens: 2,
      }),
    });
  });

  it("turns malformed function arguments into tool result errors", () => {
    const response = {
      id: "resp_1",
      output: [
        {
          type: "function_call",
          call_id: "call_1",
          name: "shell",
          arguments: '{"command":',
        },
      ],
    } as unknown as Response;

    expect(responseToolCalls(response)).toEqual([]);
    expect(responseToLinguaEvents(response)).toContainEqual({
      type: "tool_result",
      tool_call_id: "call_1",
      result: {
        ok: false,
        error: expect.stringContaining("Invalid JSON arguments for shell"),
      },
    });
  });
});

describe("tool-call argument replay", () => {
  // lingua stores arguments behind a {type:"valid",value} validation tag. The
  // model never sent that tag and must never see it: shown its own past call
  // in that shape, a model copies the wrapper, gets "missing field", sees the
  // failure replayed, and nests deeper - the runaway that burned ~$9 over 236
  // rounds. The Responses path is clean because lingua unwraps its own tag;
  // the chat-completions path (OpenRouter, where that runaway happened) is
  // hand-rolled here and has to unwrap explicitly.
  const wrapped = [
    {
      role: "assistant",
      content: [
        {
          type: "tool_call",
          tool_call_id: "call_1",
          tool_name: "shell",
          arguments: { type: "valid", value: { command: "echo hi" } },
        },
      ],
    },
  ] as never;

  it("strips the validation wrapper on the chat-completions path", () => {
    const [message] = messagesToChatMessages(wrapped);
    const toolCalls = (
      message as { tool_calls?: { function: { arguments: string } }[] }
    ).tool_calls;

    expect(toolCalls).toHaveLength(1);
    expect(toolCalls?.[0].function.arguments).toBe('{"command":"echo hi"}');
    expect(toolCalls?.[0].function.arguments).not.toContain("valid");
  });

  it("strips the validation wrapper on the responses path", () => {
    const [item] = linguaMessagesToResponsesInput(wrapped) as {
      arguments?: string;
    }[];

    expect(item.arguments).toBe('{"command":"echo hi"}');
  });
});

describe("user message image parts", () => {
  // This module is shared by every harness in the repo, several of which are
  // untouched benchmark baselines, so the image branch must be unreachable
  // for content that carries no image part.
  it("leaves ordinary user content as a plain string", () => {
    const [message] = messagesToChatMessages([
      { role: "user", content: "just text" },
    ] as never);

    expect(message).toEqual({ role: "user", content: "just text" });
  });

  it("keeps flattening a part array that has no image in it", () => {
    const [message] = messagesToChatMessages([
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ] as never);

    expect(typeof (message as { content: unknown }).content).toBe("string");
  });

  it("sends raw base64 as a data URL built from media_type", () => {
    const [message] = messagesToChatMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "look:" },
          { type: "image", image: "QUJD", media_type: "image/png" },
        ],
      },
    ] as never);

    expect(message).toEqual({
      role: "user",
      content: [
        { type: "text", text: "look:" },
        {
          type: "image_url",
          image_url: { url: "data:image/png;base64,QUJD" },
        },
      ],
    });
  });

  it("passes an http or data URL through untouched", () => {
    const [message] = messagesToChatMessages([
      {
        role: "user",
        content: [
          { type: "image", image: "https://example.com/a.png" },
          { type: "image", image: "data:image/gif;base64,R0lG" },
        ],
      },
    ] as never);

    expect((message as { content: unknown[] }).content).toEqual([
      { type: "image_url", image_url: { url: "https://example.com/a.png" } },
      { type: "image_url", image_url: { url: "data:image/gif;base64,R0lG" } },
    ]);
  });

  it("falls back to image/jpeg when media_type is absent", () => {
    const [message] = messagesToChatMessages([
      { role: "user", content: [{ type: "image", image: "QUJD" }] },
    ] as never);

    expect(
      (message as { content: { image_url: { url: string } }[] }).content[0]
        .image_url.url,
    ).toBe("data:image/jpeg;base64,QUJD");
  });
});
