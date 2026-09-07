// Context compaction for the pi-core harness, wired into pi-agent-core's own
// `transformContext` seam.
//
// Why this exists: measured on 2026-09-07, an 8-turn conversation grew the
// prompt linearly and without any ceiling - ~2,980 tokens/turn on the default
// Exo harness, ~810/turn here. Neither harness has a compaction mechanism, so
// nothing stops that line: extrapolated, the default harness reaches a 200K
// context around turn 65-70 and then simply starts failing. pi-agent-core
// ships a real compaction module and Exo's integration was not using a line
// of it.
//
// What is borrowed from pi and what is not:
//
//   - The *decision* is pi's own: estimateContextTokens() and shouldCompact()
//     with DEFAULT_COMPACTION_SETTINGS, run against the real model context
//     window. No hand-rolled threshold.
//   - The *reduction* is local. pi's compact()/prepareCompaction()/
//     findCutPoint() operate on Entry[] (its session model, which this
//     integration does not use - Exo's Rust event log is the transcript of
//     record) and its summarizer needs pi-ai's `Models` provider dispatch,
//     which createExoStreamFn deliberately bypasses to keep Exo's own model
//     routing, secrets and cost accounting. Reaching those would mean
//     adopting pi's session layer wholesale, which is a much larger change
//     than the overflow this is meant to prevent.
//
// So the reduction here is deterministic and LLM-free: keep the opening user
// message (the task anchor - dropping it loses what the agent was asked to
// do) plus the most recent messages inside pi's own keepRecentTokens budget,
// and replace the dropped middle with one marker message saying what went.
//
// pi also ships createFileOps/extractFileOpsFromMessage/computeFileLists/
// formatFileOperations, which would let the marker name the files the dropped
// span touched instead of just counting messages. They are not used because
// they are not reachable: the package's exports map exposes only ".",
// "./node" and "./session/testing", and those helpers live in
// harness/compaction/utils, so a deep import would be rejected at runtime.
//
// Deterministic beats LLM-summarised here for a second reason: a summary call
// is another model round-trip that can itself fail, hang, or cost money at
// exactly the moment the context is already in trouble.

import {
  DEFAULT_COMPACTION_SETTINGS,
  estimateContextTokens,
  estimateTokens,
  shouldCompact,
  type AgentMessage,
} from "@earendil-works/pi-agent-core";

export interface CompactionOutcome {
  compacted: boolean;
  tokensBefore: number;
  tokensAfter: number;
  droppedMessages: number;
  contextWindow: number;
}

export interface CompactContextOptions {
  contextWindow: number;
  settings?: typeof DEFAULT_COMPACTION_SETTINGS;
  /** Reports what happened, for the event log. Never throws into the turn. */
  onOutcome?: (outcome: CompactionOutcome) => void;
}

// A context window of 0 means "unknown" (buildModelStub's old default). Nothing
// can be decided from it, so compaction stays off rather than guessing a
// window and truncating a conversation that was fine.
export function compactAgentContext(
  messages: AgentMessage[],
  options: CompactContextOptions,
): AgentMessage[] {
  const settings = options.settings ?? DEFAULT_COMPACTION_SETTINGS;
  if (!settings.enabled || options.contextWindow <= 0) {
    return messages;
  }

  const before = estimateContextTokens(messages).tokens;
  if (!shouldCompact(before, options.contextWindow, settings)) {
    return messages;
  }

  const kept = keepWithinBudget(messages, settings.keepRecentTokens);
  if (kept === null || kept.droppedCount === 0) {
    // Everything left is already inside the recent budget - the context is
    // over the threshold on its own tail, and dropping more would delete the
    // messages the model is actively working from.
    return messages;
  }

  const result = [...kept.head, kept.marker, ...kept.tail];
  options.onOutcome?.({
    compacted: true,
    tokensBefore: before,
    tokensAfter: estimateContextTokens(result).tokens,
    droppedMessages: kept.droppedCount,
    contextWindow: options.contextWindow,
  });
  return result;
}

function keepWithinBudget(
  messages: AgentMessage[],
  keepRecentTokens: number,
): {
  head: AgentMessage[];
  tail: AgentMessage[];
  marker: AgentMessage;
  droppedCount: number;
} | null {
  // The first user message is the task anchor and is always kept: everything
  // after it is elaboration, but losing it loses the goal itself.
  const anchorIndex = messages.findIndex((message) => message.role === "user");
  const head = anchorIndex >= 0 ? messages.slice(0, anchorIndex + 1) : [];

  // Walk backwards accumulating pi's own per-message estimate until the
  // recent budget is spent. Stops at the anchor so head and tail can't
  // overlap.
  let cut = messages.length;
  let tailTokens = 0;
  for (let i = messages.length - 1; i > anchorIndex; i -= 1) {
    const cost = estimateTokens(messages[i]);
    if (tailTokens + cost > keepRecentTokens && cut < messages.length) {
      break;
    }
    tailTokens += cost;
    cut = i;
  }

  // Then move the cut forward to the next turn boundary. Cutting at an
  // arbitrary index orphans a tool result from the assistant tool_call it
  // answers, and a "tool" role message with no matching call ahead of it is
  // not a legal sequence - the provider rejects the whole request. Measured:
  // the first live run of this compaction cut mid-turn and every following
  // turn came back "[turn ended: model call failed]". This is the same
  // sequencing rule that broke piMessageToExoMessage once before (see this
  // package's header), and the reason pi's own findCutPoint tracks turn
  // starts rather than raw indices.
  // No later turn boundary means the budget point sits inside the newest
  // turn, and there is no legal place to cut: keeping nothing after the
  // marker deletes the work in progress, which is exactly what the first
  // live run did - the request went out as developer + anchor + marker with
  // no tool results at all and the provider answered 400. Refusing to
  // compact is the correct outcome; the turn stays large but valid.
  const safeCut = nextTurnBoundary(messages, cut);
  if (safeCut === null) {
    return null;
  }
  const tail = messages.slice(safeCut);

  const droppedStart = head.length;
  const droppedEnd = messages.length - tail.length;
  const dropped = messages.slice(droppedStart, droppedEnd);

  return {
    head,
    tail,
    droppedCount: dropped.length,
    marker: {
      role: "user",
      content:
        `[context compacted] ${dropped.length} earlier message(s) were removed to stay ` +
        `inside the model's context window. The task above and the most recent work below ` +
        `are intact; anything else from the middle of this conversation is gone, so re-read ` +
        `files or re-run commands rather than relying on remembering them.`,
      timestamp: Date.now(),
    } as AgentMessage,
  };
}

// A user message starts a turn: everything a tool_call/tool_result pair needs
// sits after one and before the next. Returns the first such index at or
// after `from`, or null when there is none - in which case the caller keeps
// the conversation whole rather than cutting somewhere illegal.
function nextTurnBoundary(
  messages: AgentMessage[],
  from: number,
): number | null {
  for (let i = from; i < messages.length; i += 1) {
    if (messages[i].role === "user") {
      return i;
    }
  }
  return null;
}
