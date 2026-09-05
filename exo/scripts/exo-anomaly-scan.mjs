#!/usr/bin/env node

// Scans the local .exo event store for the failure signatures that cost real
// money before anyone noticed them, and prints a report.
//
// Written after a single benchmark case burned ~$9 of OpenRouter credit in one
// conversation: a model double-encoded its tool-call arguments, got back an
// unrelated "missing field `command`" error, and resent the identical call 230
// times over 2.5 hours. Nothing watched for that. Every signature below is one
// that incident (or its neighbours) actually produced, not a hypothetical:
//
//   runaway-rounds      a conversation with far more model round-trips than a
//                       turn needs (the incident made 236; the busiest healthy
//                       benchmark case made 19). Counts `messages` events
//                       specifically, not raw events: pi-harness conversations
//                       legitimately hold ~1000 sandbox_process_event log lines
//                       against 4 tool calls, so raw event count flags them as
//                       runaway when nothing is wrong. One `messages` event is
//                       one paid model call, which is what actually burns money.
//   tool-error-streak   consecutive failing tool calls - the thing that was
//                       actually pathological, independent of event count
//   breaker-fired       the consecutive-tool-error circuit breaker aborted a
//                       turn (turn-loop.ts / harness-pi-core.ts write these);
//                       the loop was stopped, but something still made a model
//                       fail repeatedly and that is worth reading
//   unfinished-turn     a turn_started with no turn_ended long after the fact -
//                       the hung-process signature (one CLI process sat idle
//                       for 8 hours this way)
//
// Read-only: it never writes to the store, so it is safe to run on a live
// instance and safe to run from a scheduled task.
//
// Usage:
//   node exo/scripts/exo-anomaly-scan.mjs [--root .exo] [--since-hours 24] [--json]

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

// Model round-trips in one conversation. The busiest healthy benchmark case
// observed made 19; the runaway made 236. 50 leaves real headroom for a hard
// task while still catching a loop long before it costs dollars.
const RUNAWAY_ROUND_COUNT = 50;
// Matches MAX_CONSECUTIVE_TOOL_ERRORS in exoharness/typescript/harness/index.ts.
// A streak at or above it means the breaker either fired or would have.
const TOOL_ERROR_STREAK = 5;
// A turn still open this long after its last event is not "slow", it is stuck:
// the longest legitimate benchmark case observed ran under 4 minutes.
const UNFINISHED_TURN_MINUTES = 30;

const options = parseArguments(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const findings = await scan(options);

if (options.json) {
  console.log(JSON.stringify({ findings }, null, 2));
} else {
  printReport(findings, options);
}

// Exit code is advisory: a scheduled task reads the report either way, but a
// human running this by hand gets a shell-usable signal.
process.exit(findings.length > 0 ? 1 : 0);

async function scan({ root, sinceHours }) {
  const cutoffMs = Date.now() - sinceHours * 60 * 60 * 1000;
  const agentsDirectory = path.join(root, "exoharness", "agents");
  const findings = [];

  for (const agentId of await listDirectories(agentsDirectory)) {
    const conversationsDirectory = path.join(
      agentsDirectory,
      agentId,
      "conversations",
    );
    for (const conversationId of await listDirectories(
      conversationsDirectory,
    )) {
      const eventsDirectory = path.join(
        conversationsDirectory,
        conversationId,
        "events",
      );
      const eventFiles = (await listFiles(eventsDirectory)).filter((name) =>
        name.endsWith(".json"),
      );
      if (eventFiles.length === 0) {
        continue;
      }

      const events = await readEvents(eventsDirectory, eventFiles);
      if (events.length === 0) {
        continue;
      }
      // Skip conversations that finished before the window - a two-week-old
      // runaway is history, not something to wake anyone about.
      const lastEventMs = events[events.length - 1].createdAtMs;
      if (Number.isFinite(lastEventMs) && lastEventMs < cutoffMs) {
        continue;
      }

      const where = { agentId, conversationId };
      findings.push(
        ...runawayRoundFindings(where, events),
        ...toolErrorStreakFindings(where, events),
        ...breakerFiredFindings(where, events),
        ...unfinishedTurnFindings(where, events),
      );
    }
  }

  return findings;
}

function runawayRoundFindings(where, events) {
  const rounds = events.filter(
    (event) => event.data?.type === "messages",
  ).length;
  if (rounds < RUNAWAY_ROUND_COUNT) {
    return [];
  }
  return [
    {
      kind: "runaway-rounds",
      ...where,
      detail: `${rounds} model round-trips in one conversation (threshold ${RUNAWAY_ROUND_COUNT})`,
      rounds,
    },
  ];
}

// The signature that actually mattered: not "many rounds" but "many rounds that
// all failed the same way". Reported with the error text so whoever reads the
// report can tell a model quirk from a broken tool.
function toolErrorStreakFindings(where, events) {
  let streak = 0;
  let longest = 0;
  let lastError = null;
  let longestError = null;

  for (const event of events) {
    if (event.data?.type !== "tool_result") {
      continue;
    }
    if (isErrorResult(event.data.result)) {
      streak += 1;
      lastError = errorText(event.data.result) ?? lastError;
      if (streak > longest) {
        longest = streak;
        longestError = lastError;
      }
    } else {
      streak = 0;
      lastError = null;
    }
  }

  if (longest < TOOL_ERROR_STREAK) {
    return [];
  }
  return [
    {
      kind: "tool-error-streak",
      ...where,
      detail: `${longest} consecutive failing tool calls (threshold ${TOOL_ERROR_STREAK})${
        longestError ? `: ${truncate(longestError, 200)}` : ""
      }`,
      streak: longest,
    },
  ];
}

function breakerFiredFindings(where, events) {
  const fired = events.filter(
    (event) =>
      event.data?.type === "artifact_written" &&
      typeof event.data.path === "string" &&
      event.data.path.endsWith("aborted-consecutive-tool-errors.json"),
  );
  return fired.map((event) => ({
    kind: "breaker-fired",
    ...where,
    detail: `consecutive-tool-error breaker aborted a turn (${event.data.path})`,
  }));
}

function unfinishedTurnFindings(where, events) {
  const openTurns = new Map();
  for (const event of events) {
    if (!event.turnId) {
      continue;
    }
    if (event.data?.type === "turn_started") {
      openTurns.set(event.turnId, event.createdAtMs);
    } else if (event.data?.type === "turn_ended") {
      openTurns.delete(event.turnId);
    }
  }
  if (openTurns.size === 0) {
    return [];
  }

  const lastEventMs = events[events.length - 1].createdAtMs;
  const staleMs = UNFINISHED_TURN_MINUTES * 60 * 1000;
  const findings = [];
  for (const [turnId, startedMs] of openTurns) {
    // Measure staleness from the conversation's own last event, not from now:
    // a turn whose events stopped an hour ago is stuck, while one still
    // emitting events is merely slow.
    const idleMs = lastEventMs - startedMs;
    if (idleMs < staleMs) {
      continue;
    }
    findings.push({
      kind: "unfinished-turn",
      ...where,
      turnId,
      detail: `turn started ${Math.round(idleMs / 60000)}m before the conversation's last event and never ended`,
    });
  }
  return findings;
}

// Every tool call that goes through ToolRegistry.executePending normalizes a
// thrown error to {ok: false, error: ...} (exoharness/typescript/harness/tools.ts).
function isErrorResult(result) {
  return isRecord(result) && result.ok === false;
}

function errorText(result) {
  return isRecord(result) && typeof result.error === "string"
    ? result.error
    : null;
}

async function readEvents(directory, fileNames) {
  const events = [];
  for (const fileName of fileNames.sort()) {
    try {
      const raw = await fs.readFile(path.join(directory, fileName), "utf8");
      const parsed = JSON.parse(raw);
      events.push({
        turnId: parsed.turn_id ?? null,
        createdAtMs: Date.parse(parsed.created_at ?? ""),
        data: parsed.data ?? null,
      });
    } catch {
      // A half-written or unreadable event must not abort the whole scan -
      // this runs against a live store where writes happen concurrently.
    }
  }
  return events;
}

async function listDirectories(directory) {
  return (await readDirectory(directory))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
}

async function listFiles(directory) {
  return (await readDirectory(directory))
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name);
}

async function readDirectory(directory) {
  try {
    return await fs.readdir(directory, { withFileTypes: true });
  } catch {
    return [];
  }
}

function printReport(findings, { root, sinceHours }) {
  const header = `exo anomaly scan: root=${root} window=${sinceHours}h`;
  if (findings.length === 0) {
    console.log(`${header}\nNo anomalies found.`);
    return;
  }
  console.log(`${header}\n${findings.length} finding(s):\n`);
  for (const finding of findings) {
    console.log(`[${finding.kind}] agent=${finding.agentId}`);
    console.log(`  conversation=${finding.conversationId}`);
    console.log(`  ${finding.detail}\n`);
  }
}

function parseArguments(argv) {
  const options = {
    root: ".exo",
    sinceHours: 24,
    json: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      options.help = true;
    } else if (argument === "--json") {
      options.json = true;
    } else if (argument === "--root") {
      options.root = argv[(index += 1)] ?? options.root;
    } else if (argument === "--since-hours") {
      const value = Number(argv[(index += 1)]);
      if (Number.isFinite(value) && value > 0) {
        options.sinceHours = value;
      }
    }
  }
  return options;
}

function printHelp() {
  console.log(
    [
      "Usage: node exo/scripts/exo-anomaly-scan.mjs [options]",
      "",
      "Options:",
      "  --root <path>          event store root (default: .exo)",
      "  --since-hours <n>      only report conversations active in this window (default: 24)",
      "  --json                 emit findings as JSON",
      "  -h, --help             show this help",
      "",
      "Exits 1 when anomalies are found, 0 when clean.",
    ].join("\n"),
  );
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function truncate(text, limit) {
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}
