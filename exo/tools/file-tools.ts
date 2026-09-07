// File tools for the pi-core harness: write / edit / read.
//
// Why these exist at all: Exo's built-in tool set is shell-only
// (exoharness/typescript/harness/built-in-tools.ts's BuiltInToolName is
// "shell" | "inspect_tools" | "manage_tool" | install/uninstall), so every
// file write has to be expressed as a shell command string. Live benchmarking
// on openai/gpt-5-nano showed that is where long tasks actually break: the
// model has to get JSON string escaping *and* bash quoting right at the same
// time, and a heredoc opened inside an already single-quoted `bash -lc '...'`
// wrapper collapses the quoting (confirmed in bench-t5-multifile-r3, where
// `<<'SH'` inside `bash -lc '...'` made $1/$2 expand at top level and the
// script died with "operand expected"). Real Pi does not have this failure
// mode because it ships dedicated read/write/edit tools whose content travels
// as a plain JSON string field (dist/core/tools/write.js), never through a
// shell parser.
//
// The fix is the same idea, adapted to Exo's architecture: the model passes
// content verbatim, and *this module* - not the model - does the escaping,
// deterministically, by base64-encoding on the way in and decoding inside the
// sandbox. There is no quoting decision left for the model to get wrong.
//
// Registered by the pi-core harness (exo/harness-pi-core.ts, default as of
// its file-tools benchmark results) and its explicit -files alias, never by
// registerBuiltInTools - the default "exo" harness (exo/harness.ts) and the
// real-Pi-CLI wrapper (examples/typescript/pi-harness.ts) are untouched, so
// those baselines stay exactly as they are for comparison.

import type {
  HarnessToolRegistry,
  JsonObject,
  ToolExecutionContext,
  ToolInstance,
  ToolResult,
} from "@exo/harness";

// Mirrors pi's own read limits (dist/core/tools/truncate.js).
const MAX_READ_LINES = 2_000;
const MAX_READ_BYTES = 50 * 1024;

// Pi ships each tool's usage guidance with the tool itself (promptSnippet /
// promptGuidelines in dist/core/tools/*.js, assembled by buildSystemPrompt),
// which is what actually steers the model away from shell for file work. A
// tool the model is never told to prefer stays unused: the first live run
// with these registered still opened with four shell calls before reaching
// for edit. So the guidance ships alongside the registration - it is part of
// the design being tested, not a separate variable.
// The "no echo/printf redirection" clause was measured live to overreach:
// gpt-5-nano generalized it from "don't author file content that way" to
// "don't use echo/cat for anything," including the self-check step at the
// end of an otherwise-plain shell command (`echo $((n+1)) > f` with no
// trailing `cat f` to confirm the new value). Across repeated single-
// increment shell calls that made it lose count silently, where an
// identically-tooled native-harness agent without this wording kept a
// trailing `cat` and tracked the count correctly. The fix scopes the ban to
// authoring content and says explicitly that verifying a command's own
// result is unaffected.
export const FILE_TOOLS_INSTRUCTION =
  "File tools: use write to create a file or replace one entirely, and edit for targeted changes to an existing file. Pass file content verbatim in those tools' arguments. Do not author a file's contents through shell (no heredocs, no echo/printf redirection used to write out text): shell quoting inside a tool argument is error-prone and write/edit avoid it entirely. This is only about authoring content - keep using shell to run commands, redirect a command's own computed output, and check results with echo/cat/tail as normal.";

export function registerFileTools(tools: HarnessToolRegistry): void {
  tools.register(writeToolInstance());
  tools.register(editToolInstance());
  tools.register(readToolInstance());
}

export function writeToolInstance(): ToolInstance {
  return {
    source: "built_in",
    definition: {
      name: "write",
      description:
        "Write a file inside the sandbox, creating parent directories and overwriting an existing file. Pass the file body verbatim in content: do not wrap it in quotes, do not escape it, and do not use a heredoc. Prefer this over shell for creating or fully rewriting a file.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          path: {
            type: "string",
            description: "Absolute path of the file to write in the sandbox.",
          },
          content: {
            type: "string",
            description: "Exact file body, written as-is.",
          },
        },
        required: ["path", "content"],
      },
    },
    handler: {
      async execute(args, execution): Promise<ToolResult> {
        rejectUnknownArguments(args, ["path", "content"]);
        const path = requireString(args, "path");
        const content = requireString(args, "content");
        await writeFile(execution, path, content);
        return {
          path,
          bytes_written: Buffer.byteLength(content, "utf8"),
        };
      },
    },
  };
}

export function editToolInstance(): ToolInstance {
  return {
    source: "built_in",
    definition: {
      name: "edit",
      description:
        "Edit a file inside the sandbox by exact text replacement. Every edits[].old_text must appear exactly once in the file; if it appears zero times or more than once the edit is rejected and the file is left untouched. Pass old_text and new_text verbatim, without quoting or escaping.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          path: {
            type: "string",
            description: "Absolute path of the file to edit in the sandbox.",
          },
          edits: {
            type: "array",
            description: "Replacements applied in order.",
            items: {
              type: "object",
              additionalProperties: false,
              properties: {
                old_text: {
                  type: "string",
                  description:
                    "Exact existing text, must be unique in the file.",
                },
                new_text: {
                  type: "string",
                  description: "Replacement text.",
                },
              },
              required: ["old_text", "new_text"],
            },
          },
        },
        required: ["path", "edits"],
      },
    },
    handler: {
      async execute(args, execution): Promise<ToolResult> {
        rejectUnknownArguments(args, ["path", "edits"]);
        const path = requireString(args, "path");
        const edits = parseEdits(args.edits);
        let content = await readFile(execution, path);
        for (const [index, edit] of edits.entries()) {
          const occurrences = countOccurrences(content, edit.oldText);
          if (occurrences !== 1) {
            // Reject the whole call rather than write a partial edit: the file
            // on disk stays as it was, so the model can re-read and retry
            // against known state.
            throw new Error(
              `edits[${index}].old_text matches ${occurrences} times in ${path} (must match exactly once); file left unchanged`,
            );
          }
          content = content.replace(edit.oldText, edit.newText);
        }
        await writeFile(execution, path, content);
        return { path, edits_applied: edits.length };
      },
    },
  };
}

export function readToolInstance(): ToolInstance {
  return {
    source: "built_in",
    definition: {
      name: "read",
      description:
        "Read a file from the sandbox. Long files are truncated; when that happens the result says how many lines remain and which offset to pass to continue from there. Use offset/limit to page through a file instead of re-reading it whole - this is also how to recover the part of a large shell result that was cut off, after writing it to a file.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          path: {
            type: "string",
            description: "Absolute path of the file to read in the sandbox.",
          },
          // Strict mode has no optional properties: every key must be in
          // `required`, so "not provided" has to be expressed as null.
          offset: {
            type: ["integer", "null"],
            description:
              "1-indexed line to start reading from. Null reads from the first line.",
          },
          limit: {
            type: ["integer", "null"],
            description:
              "Maximum number of lines to return. Null reads as many as the size limit allows.",
          },
        },
        required: ["path", "offset", "limit"],
      },
    },
    handler: {
      async execute(args, execution): Promise<ToolResult> {
        rejectUnknownArguments(args, ["path", "offset", "limit"]);
        const path = requireString(args, "path");
        const offset = optionalPositiveInteger(args, "offset");
        const limit = optionalPositiveInteger(args, "limit");
        const content = await readFile(execution, path);
        const page = readPage(content, offset, limit);
        return {
          path,
          content: page.text,
          truncated: page.truncated,
          // Only present when there is actually more to read, so its absence
          // means "this is the whole file from here" rather than "unknown".
          ...(page.nextOffset === null
            ? {}
            : {
                remaining_lines: page.remainingLines,
                next_offset: page.nextOffset,
              }),
        };
      },
    },
  };
}

// Paging exists so a cut-off read has a way forward that is not "read the
// whole file again". It is also the recovery path for a truncated shell
// result: built-in-tools.ts spills the full output to a file in the sandbox,
// and this is what reads the rest of it. Before this, the two halves of that
// story used different mechanisms - the shell result pointed at a file that
// `read` could only return the first 2,000 lines of - so a model had to fall
// back to `shell tail/sed` to see anything past that.
//
// Matches pi's own read tool convention (1-indexed offset, a trailing note
// naming the next offset) so the two behave the same way where they overlap.
function readPage(
  content: string,
  offset: number | undefined,
  limit: number | undefined,
): {
  text: string;
  truncated: boolean;
  remainingLines: number;
  nextOffset: number | null;
} {
  const lines = content.split("\n");
  const start = Math.min((offset ?? 1) - 1, lines.length);
  const windowed = lines.slice(
    start,
    limit === undefined ? undefined : start + limit,
  );

  // The size cap still applies inside the window: a caller can ask for a
  // million lines, and the result still has to fit in a tool response.
  const capped = truncate(windowed.join("\n"));
  const returnedLines =
    capped.text.length === 0 ? 0 : capped.text.split("\n").length;
  const consumedThrough = start + returnedLines;
  const remainingLines = Math.max(lines.length - consumedThrough, 0);

  return {
    text: capped.text,
    truncated: capped.truncated || remainingLines > 0,
    remainingLines,
    nextOffset: remainingLines > 0 ? consumedThrough + 1 : null,
  };
}

function optionalPositiveInteger(
  args: JsonObject,
  key: string,
): number | undefined {
  const value = args[key];
  // null is how strict mode says "not provided", so it is not an error.
  if (value === null || value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new Error(`${key} must be a positive integer or null`);
  }
  return value;
}

// --- sandbox filesystem access -------------------------------------------
//
// Tools cannot touch the sandbox filesystem directly: the only seam is
// context.executeTool({functionName: "shell"}), which the Rust core runs in
// the sandbox. Content therefore travels as base64 in both directions, which
// is exactly what keeps it out of the shell's parser.

async function writeFile(
  execution: ToolExecutionContext,
  path: string,
  content: string,
): Promise<void> {
  const encoded = Buffer.from(content, "utf8").toString("base64");
  const quotedPath = shellQuote(path);
  const command =
    `mkdir -p -- "$(dirname -- ${quotedPath})" && ` +
    `printf %s ${shellQuote(encoded)} | base64 -d > ${quotedPath}`;
  const outcome = await runShell(execution, command);
  if (outcome.exitCode !== 0) {
    throw new Error(
      `write failed for ${path} (exit ${outcome.exitCode}): ${outcome.stderr.trim() || "no stderr"}`,
    );
  }
}

async function readFile(
  execution: ToolExecutionContext,
  path: string,
): Promise<string> {
  const quotedPath = shellQuote(path);
  // Without pipefail, a pipeline's exit code is the *last* command's - and
  // `tr` succeeds even on empty input, so a failing `base64` (e.g. missing
  // file) was silently swallowed as an empty read instead of an error. This
  // is exactly what happened in the first live Case 1 RSI run: the model
  // read a "" body for a file that actually had content, and then
  // overwrote it thinking it was empty.
  const command = `set -o pipefail && base64 < ${quotedPath} | tr -d '\\n'`;
  const outcome = await runShell(execution, command);
  if (outcome.exitCode !== 0) {
    throw new Error(
      `read failed for ${path} (exit ${outcome.exitCode}): ${outcome.stderr.trim() || "no stderr"}`,
    );
  }
  return Buffer.from(outcome.stdout.trim(), "base64").toString("utf8");
}

interface ShellOutcome {
  exitCode: number;
  stdout: string;
  stderr: string;
}

async function runShell(
  execution: ToolExecutionContext,
  command: string,
): Promise<ShellOutcome> {
  const result = await execution.context.executeTool({
    functionName: "shell",
    arguments: { command },
  });
  return normalizeShellResult(result);
}

// The shell tool's result shape is owned by the Rust core, and different
// call paths have been seen wrapping it (value / details). Read whichever
// layer actually carries exit_code rather than assuming one.
function normalizeShellResult(result: ToolResult): ShellOutcome {
  const outer = asRecord(result);
  const inner =
    findShellRecord(outer) ??
    findShellRecord(asRecord(outer?.value)) ??
    findShellRecord(asRecord(outer?.details));
  if (!inner) {
    throw new Error(
      `unexpected shell result shape: ${JSON.stringify(result).slice(0, 200)}`,
    );
  }
  return {
    exitCode: typeof inner.exit_code === "number" ? inner.exit_code : 0,
    stdout: typeof inner.stdout === "string" ? inner.stdout : "",
    stderr: typeof inner.stderr === "string" ? inner.stderr : "",
  };
}

function findShellRecord(
  record: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (!record) {
    return null;
  }
  return "exit_code" in record || "stdout" in record ? record : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

// POSIX single-quoting: everything inside '...' is literal, and an embedded
// quote is closed, escaped, and reopened. This is the one escaping decision
// in the whole path, and it lives here instead of in the model's head.
function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

// --- argument handling ----------------------------------------------------

interface FileEdit {
  oldText: string;
  newText: string;
}

// Mirrors built-in-tools.ts's rejectUnknownArguments: the JSON schema already
// declares additionalProperties: false, but that only constrains a model that
// generates arguments against the schema in the first place. The incident
// that motivated this file's other hardening (see readFile's pipefail
// comment, and the history-replay fix in harness/index.ts) was exactly a
// case of a malformed argument shape reaching a tool anyway - this is the
// same defense-in-depth built-in tools already have, applied here too.
function rejectUnknownArguments(args: JsonObject, allowed: string[]): void {
  const unknown = Object.keys(args).find((key) => !allowed.includes(key));
  if (unknown) {
    throw new Error(`tool argument ${unknown} is not allowed`);
  }
}

function requireString(args: JsonObject, key: string): string {
  const value = args[key];
  if (typeof value !== "string") {
    throw new Error(`${key} must be a string`);
  }
  return value;
}

function parseEdits(value: unknown): FileEdit[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("edits must be a non-empty array");
  }
  return value.map((entry, index) => {
    const record = asRecord(entry);
    const oldText = record?.old_text;
    const newText = record?.new_text;
    if (typeof oldText !== "string" || typeof newText !== "string") {
      throw new Error(
        `edits[${index}] must have string old_text and new_text fields`,
      );
    }
    return { oldText, newText };
  });
}

function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) {
    return 0;
  }
  let count = 0;
  let index = haystack.indexOf(needle);
  while (index !== -1) {
    count += 1;
    index = haystack.indexOf(needle, index + needle.length);
  }
  return count;
}

function truncate(content: string): { text: string; truncated: boolean } {
  const lines = content.split("\n");
  let text = content;
  let truncated = false;
  if (lines.length > MAX_READ_LINES) {
    text = lines.slice(0, MAX_READ_LINES).join("\n");
    truncated = true;
  }
  if (Buffer.byteLength(text, "utf8") > MAX_READ_BYTES) {
    text = Buffer.from(text, "utf8")
      .subarray(0, MAX_READ_BYTES)
      .toString("utf8");
    truncated = true;
  }
  return { text, truncated };
}
