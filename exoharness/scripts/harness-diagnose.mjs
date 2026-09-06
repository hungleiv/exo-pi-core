#!/usr/bin/env node

// Rule-based failure classifier for a single graded conversation, in the
// spirit of AgentScope (arXiv:2609.02371) - structure a trajectory, then
// check it against named failure categories instead of asking an LLM to
// eyeball the whole raw log and guess. The paper needs an LLM judge per
// invariant because it diagnoses arbitrary open-ended tasks; a
// harness-bench.sh case has a known-shape correct trajectory (N tool calls,
// a specific final answer), so most of its categories reduce to a plain
// check over the event log - no model call, no cost.
//
// Categories (named to match the paper's taxonomy where the shape matches):
//   execution-failure       a tool call's own exit code was non-zero
//   invocation-issue        a tool call failed before it ran (bad/malformed
//                           arguments, parse error)
//   instruction-unfollowing two or more tool calls requested back to back
//                           with no result in between - a batched turn where
//                           the task demanded one call at a time
//   context-miss            the final answer matches an *earlier* tool
//                           result's value but not the *latest* one - the
//                           model answered from stale information sitting in
//                           its own context instead of what actually
//                           happened last. Found by hand this session
//                           (t3-counter: answered "4" after a 5th increment
//                           had already run and made it 5) before this
//                           script existed to catch it automatically.
//   step-loop               more tool calls than the task should ever need
//   premature-termination   fewer tool calls than the task's minimum, ending
//                           in a wrong or missing final answer
//
// Multiple candidates are kept (like the paper's "failure candidate set")
// rather than stopping at the first one found, then one is picked as
// decisive using the priority order below - a broken tool call explains a
// bad outcome more directly than a raw call-count mismatch does, so it
// outranks it when both are present.
//
// Usage:
//   node exoharness/scripts/harness-diagnose.mjs --agent <slug> \
//     --conversation <slug> --expect '<regex>' \
//     [--min-requests N] [--max-requests N] [--json]
//
// Reads via the exo CLI (conversation events), not the on-disk store
// directly: unlike exo-anomaly-scan.mjs (which sweeps every conversation and
// so reads the store in bulk for speed), this targets one already-known
// agent+conversation, and the CLI already resolves the slug->id lookup that
// would otherwise have to be reimplemented here.

import { execFileSync } from "node:child_process";
import process from "node:process";

const DECISIVE_PRIORITY = [
  "execution-failure",
  "invocation-issue",
  "instruction-unfollowing",
  "context-miss",
  "step-loop",
  "premature-termination",
];

const options = parseArguments(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const events = loadEvents(options.exo, options.agent, options.conversation);
const requests = events.filter((e) => e.data?.type === "tool_requested");
const results = events.filter((e) => e.data?.type === "tool_result");
const finalAnswer = lastAssistantText(events);
const pass = options.expect
  ? new RegExp(options.expect).test(finalAnswer)
  : null;

const candidates = [
  ...executionFailureFindings(results),
  ...invocationIssueFindings(results),
  ...instructionUnfollowingFindings(events),
  ...contextMissFindings(results, finalAnswer),
  ...(options.maxRequests != null
    ? stepLoopFindings(requests, options.maxRequests)
    : []),
  ...(options.minRequests != null && pass === false
    ? prematureTerminationFindings(requests, options.minRequests)
    : []),
];

// "Decisive" = the highest-priority kind actually present, not just the
// first one found - a broken tool call explains a bad outcome more directly
// than a raw call-count mismatch, so it outranks one even if it happened
// later in the trajectory.
const decisiveKind = DECISIVE_PRIORITY.find((kind) =>
  candidates.some((c) => c.kind === kind),
);
const decisiveFinding = candidates.find((c) => c.kind === decisiveKind);

const report = {
  agent: options.agent,
  conversation: options.conversation,
  requests: requests.length,
  finalAnswer,
  pass,
  candidates,
  decisive: decisiveFinding ?? null,
};

if (options.json) {
  console.log(JSON.stringify(report, null, 2));
} else {
  printReport(report);
}

// ---------------------------------------------------------------------------

function executionFailureFindings(results) {
  const findings = [];
  for (const event of results) {
    const shell = findShellFields(event.data.result);
    if (shell && shell.exit_code !== 0) {
      findings.push({
        kind: "execution-failure",
        detail: `exit_code ${shell.exit_code}`,
        toolCallId: event.data.tool_call_id,
      });
    }
  }
  return findings;
}

function invocationIssueFindings(results) {
  const findings = [];
  for (const event of results) {
    const result = event.data.result;
    if (isRecord(result) && (result.ok === false || result.is_error === true)) {
      findings.push({
        kind: "invocation-issue",
        detail: (result.error ?? JSON.stringify(result))
          .toString()
          .slice(0, 150),
        toolCallId: event.data.tool_call_id,
      });
    }
  }
  return findings;
}

// A "requested" run with no "result" landing before the next "requested" is
// two-plus calls the model queued in one turn without waiting - exactly the
// shape a "one call at a time" instruction forbids.
function instructionUnfollowingFindings(events) {
  const findings = [];
  let pendingSinceIndex = null;
  let batchSize = 0;
  for (const event of events) {
    const type = event.data?.type;
    if (type === "tool_requested") {
      if (pendingSinceIndex !== null) {
        batchSize += 1;
      } else {
        pendingSinceIndex = event;
        batchSize = 1;
      }
    } else if (type === "tool_result") {
      if (batchSize > 1) {
        findings.push({
          kind: "instruction-unfollowing",
          detail: `${batchSize} tool calls requested before any result came back`,
        });
      }
      pendingSinceIndex = null;
      batchSize = 0;
    }
  }
  return findings;
}

// Walks tool results in order, remembering each one whose output is a bare
// number, then checks whether the final answer matches one of the earlier
// ones instead of the last one - the model read something true, later
// events superseded it, and it answered with the stale value anyway.
function contextMissFindings(results, finalAnswer) {
  const finalNumber = soleNumber(finalAnswer);
  if (finalNumber === null) {
    return [];
  }
  const seen = [];
  for (const event of results) {
    const shell = findShellFields(event.data.result);
    const value = shell ? soleNumber(shell.stdout) : null;
    if (value !== null) {
      seen.push(value);
    }
  }
  if (seen.length === 0) {
    return [];
  }
  const latest = seen[seen.length - 1];
  if (finalNumber === latest) {
    return [];
  }
  const staleIndex = seen.lastIndexOf(finalNumber);
  if (staleIndex === -1) {
    return [];
  }
  return [
    {
      kind: "context-miss",
      detail: `answered "${finalNumber}" (seen ${seen.length - staleIndex} tool result(s) earlier) instead of the latest value "${latest}"`,
    },
  ];
}

function stepLoopFindings(requests, maxRequests) {
  if (requests.length <= maxRequests) {
    return [];
  }
  return [
    {
      kind: "step-loop",
      detail: `${requests.length} tool calls, more than the ${maxRequests} this task should ever need`,
    },
  ];
}

function prematureTerminationFindings(requests, minRequests) {
  if (requests.length >= minRequests) {
    return [];
  }
  return [
    {
      kind: "premature-termination",
      detail: `only ${requests.length} tool calls, fewer than the ${minRequests} this task requires`,
    },
  ];
}

// Handles every shape this session actually saw a shell result take: bare
// {exit_code,stdout}, nested under .value/.details (file-tools.ts's own
// normalizeShellResult has to do the same), and pi-core's
// content[0].text, which is the *object itself JSON-encoded as a string*
// rather than a nested object.
function findShellFields(node, depth = 0) {
  if (depth > 6 || node == null) {
    return null;
  }
  if (typeof node === "string") {
    const trimmed = node.trim();
    if (!trimmed.startsWith("{")) {
      return null;
    }
    try {
      return findShellFields(JSON.parse(trimmed), depth + 1);
    } catch {
      return null;
    }
  }
  if (Array.isArray(node)) {
    for (const item of node) {
      const found = findShellFields(item, depth + 1);
      if (found) {
        return found;
      }
    }
    return null;
  }
  if (isRecord(node)) {
    if (typeof node.stdout === "string" && "exit_code" in node) {
      return node;
    }
    for (const value of Object.values(node)) {
      const found = findShellFields(value, depth + 1);
      if (found) {
        return found;
      }
    }
  }
  return null;
}

// Bare-number extraction, not substring search: "51" must not match inside
// "5.10" or "invocation 5133", which a naive .includes() would.
function soleNumber(text) {
  if (typeof text !== "string") {
    return null;
  }
  const match = text.trim().match(/^(\d+)\s*$/);
  return match ? match[1] : null;
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function lastAssistantText(events) {
  let text = "";
  for (const event of events) {
    if (event.data?.type !== "messages") {
      continue;
    }
    for (const message of event.data.messages ?? []) {
      if (message.role !== "assistant") {
        continue;
      }
      const content = message.content;
      const candidate =
        typeof content === "string"
          ? content
          : Array.isArray(content)
            ? content
                .filter((part) => part?.type === "text")
                .map((part) => part.text)
                .join("")
            : "";
      if (candidate.trim()) {
        text = candidate;
      }
    }
  }
  return text.trim();
}

function loadEvents(exoPath, agent, conversation) {
  const raw = execFileSync(
    exoPath,
    ["conversation", "events", agent, conversation],
    { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
  );
  return JSON.parse(raw).events ?? [];
}

function printReport(report) {
  console.log(`agent:        ${report.agent}`);
  console.log(`conversation: ${report.conversation}`);
  console.log(`requests:     ${report.requests}`);
  console.log(`final answer: ${JSON.stringify(report.finalAnswer)}`);
  if (report.pass !== null) {
    console.log(`pass:         ${report.pass}`);
  }
  if (report.candidates.length === 0) {
    console.log("candidates:   none");
    return;
  }
  console.log(`candidates:`);
  for (const candidate of report.candidates) {
    const marker = candidate === report.decisive ? "-> " : "   ";
    console.log(`${marker}${candidate.kind}: ${candidate.detail}`);
  }
}

function parseArguments(argv) {
  const options = {
    exo: "./target/debug/exo",
    minRequests: null,
    maxRequests: null,
    expect: null,
    json: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--agent":
        options.agent = argv[++i];
        break;
      case "--conversation":
        options.conversation = argv[++i];
        break;
      case "--expect":
        options.expect = argv[++i];
        break;
      case "--min-requests":
        options.minRequests = Number(argv[++i]);
        break;
      case "--max-requests":
        options.maxRequests = Number(argv[++i]);
        break;
      case "--exo":
        options.exo = argv[++i];
        break;
      case "--json":
        options.json = true;
        break;
      case "--help":
      case "-h":
        options.help = true;
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!options.help && (!options.agent || !options.conversation)) {
    throw new Error("--agent and --conversation are required");
  }
  return options;
}

function printHelp() {
  console.log(`Usage: node exoharness/scripts/harness-diagnose.mjs --agent <slug> --conversation <slug> [options]

Options:
  --expect <regex>      Grade the final answer against this pattern
  --min-requests <n>    Flag premature-termination below this call count (only checked on fail)
  --max-requests <n>    Flag step-loop above this call count
  --exo <path>          Path to the exo binary (default ./target/debug/exo)
  --json                Print the full report as JSON
`);
}
