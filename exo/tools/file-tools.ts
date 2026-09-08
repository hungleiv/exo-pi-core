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
        // Under the same lock as edit: a write racing an edit on one path
        // loses the same way two edits do, and the queue is per path, so an
        // unlocked write would simply step over a held lock.
        await withFileLock(execution, path, () =>
          writeFile(execution, path, content),
        );
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
        return withFileLock(execution, path, async () => {
          const originalText = await readFile(execution, path);
          const lineEnding = detectLineEnding(originalText);
          // null means mixed endings: leave the bytes alone and match exactly,
          // which is what this tool did before line-ending handling existed.
          const normalize = lineEnding === null ? identity : normalizeToLF;
          // Everything below - matching, fuzzy fallback, diff line numbers -
          // works on LF text, and the original ending is put back at the end.
          let content = normalize(originalText);
          // The model reads text and writes old_text with \n even when the file
          // is CRLF, so its arguments are normalized the same way.
          const normalizedEdits = edits.map((edit) => ({
            oldText: normalize(edit.oldText),
            newText: normalize(edit.newText),
          }));
          let fuzzyMatches = 0;
          const hunks: { firstChangedLine: number; diff: string }[] = [];
          for (const [index, edit] of normalizedEdits.entries()) {
            const located = locateEdit(content, edit.oldText);
            if (located === null) {
              // Reject the whole call rather than write a partial edit: the file
              // on disk stays as it was, so the model can re-read and retry
              // against known state.
              const occurrences = countOccurrences(content, edit.oldText);
              throw new Error(
                `edits[${index}].old_text matches ${occurrences} times in ${path} (must match exactly once); file left unchanged`,
              );
            }
            if (located.fuzzy) {
              fuzzyMatches += 1;
            }
            const beforeThisEdit = content;
            content =
              content.slice(0, located.index) +
              edit.newText +
              content.slice(located.index + edit.oldText.length);
            // Per edit, not once at the end. A single before/after comparison
            // cannot tell two distant changes apart - it reports everything
            // between them as one enormous changed region. Measured: two edits
            // at lines 3 and 200 of a 300-line file produced a 401-line "diff"
            // that the cap then trimmed to 40 lines of unchanged context, with
            // the second change never visible. Each edit changes exactly one
            // contiguous span, so summarizing them one at a time is both
            // correct and cheap.
            const hunk = summarizeChange(beforeThisEdit, content);
            if (hunk !== null) {
              hunks.push(hunk);
            }
          }
          await writeFile(
            execution,
            path,
            restoreLineEndings(content, lineEnding),
          );
          const change = mergeHunks(hunks);
          return {
            path,
            edits_applied: edits.length,
            // Reported rather than silent: the file now differs from what the
            // model literally asked for at those spots, and it should be able
            // to tell that from the result instead of guessing.
            ...(fuzzyMatches > 0 ? { fuzzy_matches: fuzzyMatches } : {}),
            ...(change === null
              ? {}
              : {
                  first_changed_line: change.firstChangedLine,
                  diff: change.diff,
                }),
          };
        });
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
        "Read a file from the sandbox. Text and images (png, jpeg, gif, webp) are both supported; an image is returned as a picture you can look at, so read it rather than trying to inspect its bytes. Long text files are truncated; when that happens the result says how many lines remain and which offset to pass to continue from there. Use offset/limit to page through a file instead of re-reading it whole - this is also how to recover the part of a large shell result that was cut off, after writing it to a file.",
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
        const bytes = await readFileBytes(execution, path);

        // Images are returned as an image, not as their bytes rendered into
        // text. Decoding PNG bytes as UTF-8 produces replacement-character
        // noise that costs a lot of context and tells the model nothing, and
        // it has no way to recover from that - there is no "read this as an
        // image" argument it could retry with.
        const mediaType = detectImageMediaType(bytes);
        if (mediaType !== null) {
          if (bytes.length > MAX_READ_IMAGE_BYTES) {
            // Actionable rather than a bare refusal: the sandbox has a shell,
            // so the model can shrink the file itself and read it again.
            throw new Error(
              `${path} is a ${mediaType} image of ${formatBytes(bytes.length)}, over the ${formatBytes(MAX_READ_IMAGE_BYTES)} limit; resize it first (e.g. shell: convert or ffmpeg) and read the smaller file`,
            );
          }
          return {
            path,
            media_type: mediaType,
            bytes: bytes.length,
            // Read by the pi-core adapter, which turns it into a real image
            // content part and keeps it out of the tool-result text. See
            // pi-agent-adapters.ts's imageResultParts.
            image_base64: bytes.toString("base64"),
          };
        }

        const content = bytes.toString("utf8");
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

// Serializes mutations of the same file within one turn.
//
// Not load-bearing today, and the comment should say so rather than imply a
// live bug: harness-pi-core.ts:205 sets toolExecution "sequential", so a
// round's tool calls never overlap and the race below cannot currently
// happen. What this replaces is the *reason* that blunt setting exists. Its
// own comment (harness-pi-core.ts:150-157) gives one justification - Exo's
// tools share a sandbox filesystem and could race on the same files - and
// serializing every tool call in the harness is a very wide answer to a
// narrow hazard, paid for on every multi-call round including ones that
// touch no files at all.
//
// The hazard itself is real: `edit` is read-modify-write with an await
// between the read and the write, so under parallel execution two edits to
// one file both read the original and the second write silently discards the
// first. No error, no tool_error - the call reports success and the change is
// gone. pi handles it precisely, with withFileMutationQueue (dist/harness/
// tools/file-mutation-queue.js) keyed by canonical path, and leaves its loop
// parallel. This is the same idea, keyed by TurnContext rather than
// ExecutionEnv because that is the object that is stable across the tool
// calls of a turn here.
//
// So this exists to make lifting toolExecution "sequential" a decision that
// can be made on measured latency rather than blocked on a correctness
// worry. It does NOT lift it - that needs its own benchmark, since the other
// half of that setting's justification is shell calls racing each other,
// which a per-file lock does nothing about.
//
// Two deliberate simplifications against pi's version:
//   - the key is computed synchronously, so registration needs no separate
//     serialization step. pi's key comes from an async canonicalPath call,
//     which is why it carries a `registration` promise chain to stop two
//     callers from both observing the same predecessor; between the get and
//     the set below there is no await, so that race cannot happen here.
//   - no symlink resolution, so two paths aliasing one file through a
//     symlink still race. Resolving it costs a sandbox round-trip per edit,
//     and the failure it would prevent has not been observed; the syntactic
//     normalization below covers the case that actually shows up, which is
//     the same file addressed as /a//b and /a/./b.
const fileMutationQueues = new WeakMap<object, Map<string, Promise<void>>>();

async function withFileLock<T>(
  execution: ToolExecutionContext,
  path: string,
  operation: () => Promise<T>,
): Promise<T> {
  const scope = execution.context as unknown as object;
  let queues = fileMutationQueues.get(scope);
  if (queues === undefined) {
    queues = new Map();
    fileMutationQueues.set(scope, queues);
  }
  const key = normalizePathKey(path);

  // Nothing awaits between reading `previous` and storing `chained`, so each
  // caller observes a distinct predecessor.
  const previous = queues.get(key) ?? Promise.resolve();
  let release = (): void => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const chained = previous.then(() => held);
  queues.set(key, chained);

  // `previous` cannot reject: it is either Promise.resolve() or an earlier
  // `chained`, and every `held` is resolved from the finally below - a failing
  // operation rejects to its own caller, never into the queue.
  await previous;
  try {
    return await operation();
  } finally {
    release();
    // Only if no later caller has already chained onto this entry, otherwise
    // the next waiter loses its predecessor.
    if (queues.get(key) === chained) {
      queues.delete(key);
    }
  }
}

// Syntactic only - see the symlink note above.
function normalizePathKey(path: string): string {
  const segments: string[] = [];
  for (const segment of path.split("/")) {
    if (segment === "" || segment === ".") {
      continue;
    }
    if (segment === "..") {
      segments.pop();
      continue;
    }
    segments.push(segment);
  }
  return `${path.startsWith("/") ? "/" : ""}${segments.join("/")}`;
}

// Line-ending handling, mirroring pi's edit-diff.js. Without this an edit
// against a CRLF file fails outright: the model writes old_text with \n (it
// is reading text, not bytes), the file holds \r\n, so the match count is
// zero and the edit is rejected. normalizeTypography does not rescue it
// either - it never touches \r. Normalize on the way in, restore on the way
// out, so the file keeps the line endings it arrived with.
//
// pi also strips a BOM around its edits (stripBom in edit-diff.js). That is
// deliberately not copied: a BOM is a single leading character, so exact
// matching runs straight past it and writing the unchanged content back
// preserves it anyway. Tested by removing the strip and watching the test
// still pass - it guards nothing here, and code that guards nothing is worse
// than no code, because it implies a hazard that was handled.
//
// Stricter than pi here on one case: pi decides from the *first* line ending
// it sees and then rewrites the whole file to it (detectLineEnding +
// restoreLineEndings in edit-diff.js), so editing one line of a mixed-ending
// file silently converts every other line too. This returns null for mixed
// files instead, and the caller then matches against the raw bytes exactly as
// it did before line-ending handling existed. A mixed file is rare, but
// rewriting lines the model never asked about is the kind of silent damage
// this whole change is meant to prevent.
function detectLineEnding(content: string): "\n" | "\r\n" | null {
  if (!content.includes("\r")) {
    return "\n";
  }
  // Pure CRLF means every \r is part of a \r\n and every \n is preceded by
  // one, so removing all \r\n leaves neither behind. Anything else - a bare
  // \r, or an \n without its \r - is mixed.
  const withoutCrlf = content.replaceAll("\r\n", "");
  return withoutCrlf.includes("\r") || withoutCrlf.includes("\n")
    ? null
    : "\r\n";
}

// Only ever called on text whose file is pure LF or pure CRLF, so a bare \r
// cannot reach it - it is left out rather than handled, because converting one
// would be an unrequested change to a byte the model never mentioned.
function normalizeToLF(text: string): string {
  return text.replaceAll("\r\n", "\n");
}

function identity(text: string): string {
  return text;
}

function restoreLineEndings(
  text: string,
  ending: "\n" | "\r\n" | null,
): string {
  return ending === "\r\n" ? text.replaceAll("\n", "\r\n") : text;
}

// Base64 payload above which the single-command path is abandoned for the
// chunked one.
//
// Linux caps a *single* argv element at MAX_ARG_STRLEN, 32 pages = 131072
// bytes, independently of the much larger total ARG_MAX. The only seam into
// the sandbox is one shell command string, which arrives as `bash -lc
// <script>` - one argv element - so the whole script, base64 payload
// included, is measured against that per-element cap.
//
// Found live, not by inspection: an agent editing crates/cli/src/main.rs
// (148KB source, ~198KB base64) got "Argument list too long (os error 7)"
// and the edit was lost. Every file from roughly 96KB of content upward was
// unwritable, which is an ordinary source file, not an edge case.
//
// 64KB leaves room for the mkdir/printf/redirect boilerplate and the path
// without needing to reason about their exact lengths.
const MAX_BASE64_PER_COMMAND = 64 * 1024;

async function writeFile(
  execution: ToolExecutionContext,
  path: string,
  content: string,
): Promise<void> {
  const encoded = Buffer.from(content, "utf8").toString("base64");
  const quotedPath = shellQuote(path);
  const makeDirectory = `mkdir -p -- "$(dirname -- ${quotedPath})"`;

  if (encoded.length <= MAX_BASE64_PER_COMMAND) {
    await runWriteStep(
      execution,
      path,
      `${makeDirectory} && printf %s ${shellQuote(encoded)} | base64 -d > ${quotedPath}`,
    );
    return;
  }

  // Staged rather than appended straight onto the target: a chunk sequence
  // that fails halfway would otherwise leave a truncated file where a
  // readable one used to be. The target is replaced by a single mv at the
  // end, so it is either the old content or the new one, never half of each.
  const stagingPath = shellQuote(`${path}.exo-write.${Date.now()}`);
  const decodedPath = shellQuote(`${path}.exo-decode.${Date.now()}`);
  const cleanup = `rm -f ${stagingPath} ${decodedPath}`;

  try {
    await runWriteStep(execution, path, makeDirectory);
    for (let offset = 0; offset < encoded.length; ) {
      const chunk = encoded.slice(offset, offset + MAX_BASE64_PER_COMMAND);
      // The first chunk truncates, the rest append, so a stale staging file
      // from an interrupted earlier write cannot prepend itself to this one.
      const redirect = offset === 0 ? ">" : ">>";
      await runWriteStep(
        execution,
        path,
        `printf %s ${shellQuote(chunk)} ${redirect} ${stagingPath}`,
      );
      offset += chunk.length;
    }
    await runWriteStep(
      execution,
      path,
      `set -o pipefail && base64 -d < ${stagingPath} > ${decodedPath} && ` +
        `mv -- ${decodedPath} ${quotedPath} && rm -f ${stagingPath}`,
    );
  } catch (error) {
    // Best effort: the write already failed, and failing to tidy up must not
    // replace that error with a less informative one.
    await runShell(execution, cleanup).catch(() => {});
    throw error;
  }
}

async function runWriteStep(
  execution: ToolExecutionContext,
  path: string,
  command: string,
): Promise<void> {
  const outcome = await runShell(execution, command);
  if (outcome.exitCode !== 0) {
    throw new Error(
      `write failed for ${path} (exit ${outcome.exitCode}): ${outcome.stderr.trim() || "no stderr"}`,
    );
  }
}

// Image formats every vision-capable model behind OpenRouter's chat API
// accepts. Deliberately narrower than pi's detectSupportedImageMimeType
// (harness/tools/image.js), which also reports image/bmp - pi only emits BMP
// when an imageProcessor is configured to convert it, and there is none here,
// so claiming support would produce a request the provider rejects.
//
// Signature checks only, no decoding: the point is to tell "this is an image"
// from "this is text", not to validate the file.
function detectImageMediaType(bytes: Buffer): string | null {
  if (bytes.length >= 4 && bytes.subarray(0, 3).equals(JPEG_SIGNATURE)) {
    return "image/jpeg";
  }
  if (bytes.subarray(0, 8).equals(PNG_SIGNATURE)) {
    return "image/png";
  }
  if (bytes.subarray(0, 3).toString("ascii") === "GIF") {
    return "image/gif";
  }
  if (
    bytes.length >= 12 &&
    bytes.subarray(0, 4).toString("ascii") === "RIFF" &&
    bytes.subarray(8, 12).toString("ascii") === "WEBP"
  ) {
    return "image/webp";
  }
  return null;
}

const JPEG_SIGNATURE = Buffer.from([0xff, 0xd8, 0xff]);
const PNG_SIGNATURE = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);

// Images bypass MAX_READ_BYTES, which is a line-oriented text budget, so they
// need a budget of their own. An image costs context in proportion to its
// pixels, and there is no resizer in this harness (pi auto-resizes via an
// injected imageProcessor; nothing supplies one here), so the cap is the only
// thing standing between one screenshot and a blown context window.
const MAX_READ_IMAGE_BYTES = 3 * 1024 * 1024;

function formatBytes(count: number): string {
  return count < 1024
    ? `${count}B`
    : count < 1024 * 1024
      ? `${(count / 1024).toFixed(1)}KB`
      : `${(count / (1024 * 1024)).toFixed(1)}MB`;
}

async function readFile(
  execution: ToolExecutionContext,
  path: string,
): Promise<string> {
  return (await readFileBytes(execution, path)).toString("utf8");
}

async function readFileBytes(
  execution: ToolExecutionContext,
  path: string,
): Promise<Buffer> {
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
  return Buffer.from(outcome.stdout.trim(), "base64");
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

// Joins the per-edit summaries into one result, keeping the whole thing
// inside the size budget. Line numbers are those of the file at the moment
// each edit ran, which is what the final file reflects too - except when a
// later edit sits *before* an earlier one and changes the line count, which
// no single-pass scheme can express anyway.
function mergeHunks(
  hunks: { firstChangedLine: number; diff: string }[],
): { firstChangedLine: number; diff: string } | null {
  if (hunks.length === 0) {
    return null;
  }
  const lines: string[] = [];
  hunks.forEach((hunk, index) => {
    if (index > 0) {
      lines.push("  ...");
    }
    lines.push(...hunk.diff.split("\n"));
  });
  const capped =
    lines.length > DIFF_MAX_LINES
      ? [
          ...lines.slice(0, DIFF_MAX_LINES),
          `  ... ${lines.length - DIFF_MAX_LINES} more diff line(s) not shown`,
        ]
      : lines;
  return {
    firstChangedLine: Math.min(...hunks.map((hunk) => hunk.firstChangedLine)),
    diff: capped.join("\n"),
  };
}

// A compact before/after view of what the edit actually did.
//
// The model needs this most now that matching can be fuzzy: when
// normalization resolved the target, the bytes written differ from the
// old_text the model typed, and without seeing the result it has no way to
// notice. It also catches the plain case of an edit that landed somewhere
// other than intended.
//
// Deliberately not a full unified patch. This result goes through
// compactToolResultForModel, which cuts a tool result to ~4,000 characters -
// a whole-file diff would be truncated into uselessness on any real file,
// and would crowd out the rest of the result. Only the changed region plus a
// couple of lines of context is reported, and even that is capped.
const DIFF_CONTEXT_LINES = 2;
const DIFF_MAX_LINES = 40;

function summarizeChange(
  before: string,
  after: string,
): { firstChangedLine: number; diff: string } | null {
  if (before === after) {
    return null;
  }
  const beforeLines = before.split("\n");
  const afterLines = after.split("\n");

  // Trim the identical head and tail so only the changed span is described.
  let start = 0;
  while (
    start < beforeLines.length &&
    start < afterLines.length &&
    beforeLines[start] === afterLines[start]
  ) {
    start += 1;
  }
  let fromEnd = 0;
  while (
    fromEnd < beforeLines.length - start &&
    fromEnd < afterLines.length - start &&
    beforeLines[beforeLines.length - 1 - fromEnd] ===
      afterLines[afterLines.length - 1 - fromEnd]
  ) {
    fromEnd += 1;
  }

  const contextStart = Math.max(start - DIFF_CONTEXT_LINES, 0);
  const removed = beforeLines.slice(start, beforeLines.length - fromEnd);
  const added = afterLines.slice(start, afterLines.length - fromEnd);
  const contextBefore = beforeLines.slice(contextStart, start);
  const contextAfter = afterLines.slice(
    afterLines.length - fromEnd,
    Math.min(
      afterLines.length - fromEnd + DIFF_CONTEXT_LINES,
      afterLines.length,
    ),
  );

  const lines: string[] = [];
  contextBefore.forEach((line, i) => {
    lines.push(`  ${contextStart + i + 1} ${line}`);
  });
  removed.forEach((line, i) => {
    lines.push(`- ${start + i + 1} ${line}`);
  });
  added.forEach((line, i) => {
    lines.push(`+ ${start + i + 1} ${line}`);
  });
  contextAfter.forEach((line, i) => {
    lines.push(`  ${afterLines.length - fromEnd + i + 1} ${line}`);
  });

  return { firstChangedLine: start + 1, diff: lines.join("\n") };
}

// Finds where an edit applies, falling back to typographic normalization when
// the literal text is not there.
//
// The failure this exists for, measured on the t8-fuzzyedit benchmark case:
// a model targets a phrase containing curly quotes, an apostrophe or an
// em-dash, retypes it with the plain ASCII equivalents its own output
// naturally produces, and an exact-match edit rejects the call outright -
// 0/5 on that case, while pi-agent-core's edit (which normalizes first)
// recovered every time. The file, not the model, is the odd one out here:
// the text is "the same" to any reader.
//
// Returns null when there is no single unambiguous target, which keeps the
// existing all-or-nothing contract: ambiguity is never resolved by guessing.
function locateEdit(
  content: string,
  oldText: string,
): { index: number; fuzzy: boolean } | null {
  if (oldText.length === 0) {
    return null;
  }
  if (countOccurrences(content, oldText) === 1) {
    return { index: content.indexOf(oldText), fuzzy: false };
  }
  // More than one literal hit is genuine ambiguity - normalizing can only
  // make that worse, never better - so only a clean miss falls through here.
  if (countOccurrences(content, oldText) > 1) {
    return null;
  }

  const normalizedContent = normalizeTypography(content);
  const normalizedOld = normalizeTypography(oldText);
  if (countOccurrences(normalizedContent, normalizedOld) !== 1) {
    return null;
  }
  // Safe because normalizeTypography is strictly character-for-character:
  // every substitution is one code point for one code point, so offsets in
  // the normalized string are the same offsets in the original. That is why
  // this deliberately skips the two normalizations pi also applies - NFKC
  // and per-line trimEnd - which change length and would need the original
  // positions mapped back.
  return { index: normalizedContent.indexOf(normalizedOld), fuzzy: true };
}

// Same character classes pi's normalizeForFuzzyMatch folds, minus the two
// length-changing steps. These cover what models actually get wrong:
// retyping typographic punctuation as its ASCII lookalike.
function normalizeTypography(text: string): string {
  return text
    .replace(/[\u2018\u2019\u201A\u201B]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F]/g, '"')
    .replace(/[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]/g, "-")
    .replace(/[\u00A0\u2002-\u200A\u202F\u205F\u3000]/g, " ");
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
