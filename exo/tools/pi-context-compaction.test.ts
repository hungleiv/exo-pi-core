import { describe, expect, it } from "vitest";
import type { AgentMessage } from "@earendil-works/pi-agent-core";

import {
  compactAgentContext,
  type CompactionOutcome,
} from "./pi-context-compaction";

// ~4 chars per token is the heuristic pi's own estimateTokens uses, so this
// makes message sizes predictable without asserting on its exact internals.
function userMessage(text: string): AgentMessage {
  return { role: "user", content: text, timestamp: 0 } as AgentMessage;
}

function conversation(messageCount: number, charsEach: number): AgentMessage[] {
  const messages: AgentMessage[] = [userMessage("TASK-ANCHOR: do the thing")];
  for (let i = 0; i < messageCount; i += 1) {
    messages.push(userMessage(`m${i}:${"x".repeat(charsEach)}`));
  }
  return messages;
}

describe("compactAgentContext", () => {
  it("leaves a conversation well inside the window untouched", () => {
    const messages = conversation(5, 100);
    expect(compactAgentContext(messages, { contextWindow: 272_000 })).toBe(
      messages,
    );
  });

  // The old buildModelStub hard-coded contextWindow: 0. Compacting on an
  // unknown window would mean guessing, and a wrong guess silently deletes
  // history that was never too large.
  it("does nothing when the context window is unknown", () => {
    const messages = conversation(500, 4_000);
    expect(compactAgentContext(messages, { contextWindow: 0 })).toBe(messages);
  });

  it("is a no-op when disabled", () => {
    const messages = conversation(500, 4_000);
    expect(
      compactAgentContext(messages, {
        contextWindow: 8_000,
        settings: {
          enabled: false,
          reserveTokens: 16_384,
          keepRecentTokens: 20_000,
        },
      }),
    ).toBe(messages);
  });

  it("drops the middle but keeps the task anchor and the recent tail", () => {
    const messages = conversation(400, 4_000);
    const outcomes: CompactionOutcome[] = [];

    const result = compactAgentContext(messages, {
      contextWindow: 60_000,
      onOutcome: (outcome) => outcomes.push(outcome),
    });

    expect(result).not.toBe(messages);
    expect(result.length).toBeLessThan(messages.length);

    // The opening task survives - losing it loses what was asked.
    expect(result[0]).toBe(messages[0]);
    // The newest message survives - that is what the model is working from.
    expect(result.at(-1)).toBe(messages.at(-1));
    // Exactly one marker sits between them.
    const markers = result.filter(
      (message) =>
        typeof (message as { content?: unknown }).content === "string" &&
        String((message as { content: string }).content).startsWith(
          "[context compacted]",
        ),
    );
    expect(markers).toHaveLength(1);

    expect(outcomes).toHaveLength(1);
    expect(outcomes[0].compacted).toBe(true);
    expect(outcomes[0].droppedMessages).toBeGreaterThan(0);
    expect(outcomes[0].tokensAfter).toBeLessThan(outcomes[0].tokensBefore);
  });

  // The bug the first live run hit: cutting at an arbitrary index left a tool
  // result whose assistant tool_call had been dropped, which is not a legal
  // message sequence - every turn after that came back "[turn ended: model
  // call failed]".
  //
  // Fuzzed rather than fixed-shape on purpose. Whether a raw budget cut lands
  // somewhere illegal depends on the exact message sizes: two hand-written
  // conversations were tried first and the broken code happened to land on a
  // legal boundary in both, so the test passed against code known to be
  // wrong. Sweeping the sizes makes at least one cut land mid-pair.
  it("never orphans a tool call or its result, across many shapes", () => {
    for (let seed = 1; seed <= 60; seed += 1) {
      const messages: AgentMessage[] = [userMessage("TASK-ANCHOR")];
      for (let turn = 0; turn < 10; turn += 1) {
        messages.push(userMessage(`turn ${turn}`));
        for (let call = 0; call < 4; call += 1) {
          const id = `call_${turn}_${call}`;
          messages.push({
            role: "assistant",
            content: [
              {
                type: "toolCall",
                id,
                name: "shell",
                arguments: { command: "x" },
              },
            ],
            timestamp: 0,
          } as unknown as AgentMessage);
          messages.push({
            role: "toolResult",
            toolCallId: id,
            toolName: "shell",
            content: [{ type: "text", text: "x".repeat(1_000 + seed * 137) }],
            timestamp: 0,
          } as unknown as AgentMessage);
        }
      }

      const result = compactAgentContext(messages, { contextWindow: 30_000 });

      const calls = new Set<string>();
      const results = new Set<string>();
      for (const message of result) {
        const role = (message as { role?: string }).role;
        if (role === "assistant") {
          for (const part of (message as { content?: unknown[] }).content ??
            []) {
            const id = (part as { id?: string }).id;
            if (id) calls.add(id);
          }
        }
        if (role === "toolResult") {
          const id = (message as { toolCallId?: string }).toolCallId;
          // Every kept result must have its call kept too, and vice versa:
          // both halves of the pair or neither.
          expect(
            id && calls.has(id),
            `seed ${seed}: orphaned result ${id}`,
          ).toBe(true);
          if (id) results.add(id);
        }
      }
      for (const id of calls) {
        expect(
          results.has(id),
          `seed ${seed}: call ${id} lost its result`,
        ).toBe(true);
      }
    }
  });

  // The bug the first live run actually hit (found by dumping the real
  // requests, not by reasoning about it): when the budget point falls inside
  // the newest turn there is no later turn boundary to cut on. Returning an
  // empty tail there deletes the work in progress - the request went out as
  // developer + anchor + marker with no tool results at all, and the provider
  // answered 400 on every following turn. Refusing to compact is correct.
  it("refuses to compact when the cut would land inside the newest turn", () => {
    const messages: AgentMessage[] = [
      userMessage("TASK-ANCHOR"),
      userMessage("the only real turn"),
    ];
    for (let call = 0; call < 6; call += 1) {
      const id = `call_${call}`;
      messages.push({
        role: "assistant",
        content: [
          { type: "toolCall", id, name: "shell", arguments: { command: "x" } },
        ],
        timestamp: 0,
      } as unknown as AgentMessage);
      messages.push({
        role: "toolResult",
        toolCallId: id,
        toolName: "shell",
        content: [{ type: "text", text: "x".repeat(40_000) }],
        timestamp: 0,
      } as unknown as AgentMessage);
    }

    // Small window and small recent budget: the decision says compact, and
    // the only cut point available is mid-turn.
    const result = compactAgentContext(messages, {
      contextWindow: 20_000,
      settings: {
        enabled: true,
        reserveTokens: 100,
        keepRecentTokens: 600,
      },
    });

    expect(result).toBe(messages);
  });

  // A single enormous recent message can push the estimate over the threshold
  // on its own. There is nothing safe to drop there - the tail *is* the
  // working set - so the conversation must come back unchanged rather than
  // being cut into something the model can no longer use.
  it("does not cut when everything left is already the recent tail", () => {
    const messages = [
      userMessage("TASK-ANCHOR"),
      userMessage("y".repeat(400_000)),
    ];
    expect(compactAgentContext(messages, { contextWindow: 60_000 })).toBe(
      messages,
    );
  });
});
