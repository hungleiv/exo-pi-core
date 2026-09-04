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
// These are registered only by the pi-core harness path
// (exo/harness-pi-core-files.ts), never by registerBuiltInTools, so the
// exo-bench and pi-bench baselines stay exactly as they are.

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
export const FILE_TOOLS_INSTRUCTION =
  "File tools: use write to create a file or replace one entirely, and edit for targeted changes to an existing file. Pass file content verbatim in those tools' arguments. Do not write file content through shell (no heredocs, no echo/printf redirection): shell quoting inside a tool argument is error-prone and write/edit avoid it entirely. Keep using shell to run commands, inspect output, and test what you built.";

export function registerFileTools(tools: HarnessToolRegistry): void {
  tools.register(writeToolInstance());
  tools.register(editToolInstance());
  tools.register(readToolInstance());
}

function writeToolInstance(): ToolInstance {
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

function editToolInstance(): ToolInstance {
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

function readToolInstance(): ToolInstance {
  return {
    source: "built_in",
    definition: {
      name: "read",
      description:
        "Read a file from the sandbox. Long files are truncated; the result says so when that happens.",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          path: {
            type: "string",
            description: "Absolute path of the file to read in the sandbox.",
          },
        },
        required: ["path"],
      },
    },
    handler: {
      async execute(args, execution): Promise<ToolResult> {
        const path = requireString(args, "path");
        const content = await readFile(execution, path);
        const truncated = truncate(content);
        return {
          path,
          content: truncated.text,
          truncated: truncated.truncated,
        };
      },
    },
  };
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
  const command = `base64 < ${quotedPath} | tr -d '\\n'`;
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
