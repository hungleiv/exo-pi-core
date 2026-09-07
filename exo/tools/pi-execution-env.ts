// An ExecutionEnv backed by Exo's sandbox, so pi-agent-core's own read/write/
// edit tools can run against it instead of Exo reimplementing them.
//
// What this buys, measured against exo/tools/file-tools.ts:
//   - read takes offset/limit. Exo's does not, and that gap is why the
//     large-output work earlier this session needed a spill file at all:
//     real Pi recovers a truncated command's tail with
//     read(path, offset, limit), which had no equivalent here.
//   - read handles images (ReadImageProcessor, with auto-resize).
//   - edit returns a diff, a unified patch, and the first changed line.
//     Exo's returns edits_applied and nothing else.
//   - truncation, path resolution and the file-mutation queue come from pi's
//     tested implementations rather than from a second local copy.
//
// What it does NOT buy, contrary to the note in file-tools.ts:12-15 ("Real Pi
// does not have this failure mode because it ships dedicated read/write/edit
// tools whose content travels as a plain JSON string field, never through a
// shell parser"). That is true of real Pi because its ExecutionEnv talks to a
// real filesystem API. Exo's sandbox exposes exactly one primitive - run a
// shell command - so every method below goes back through a shell either way.
// The quoting risk is not removed, it is centralised: it now lives in these
// few functions, which are base64-framed and unit-tested, instead of being
// spread across each tool's own command construction.
//
// Only the seven methods pi's read/write/edit tools actually call are
// implemented (verified by grepping env.* out of the shipped tool sources).
// The rest of the FileSystem/Shell surface returns a "not_supported"
// FileError rather than a wrong answer, because a silently-wrong listDir or
// createTempDir is worse than an explicit refusal. Add one when a tool that
// needs it is actually wired.

import {
  ExecutionError,
  FileError,
  type ExecutionEnv,
  type FileInfo,
  type Result,
  type ShellExecOptions,
} from "@earendil-works/pi-agent-core";
import type { JsonObject, ToolResult, TurnContext } from "@exo/harness";

interface ShellOutcome {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export function createSandboxExecutionEnv(
  context: TurnContext,
  cwd = "/",
): ExecutionEnv {
  const run = (command: string) => runShell(context, command);

  const unsupported = <T>(method: string): Promise<Result<T, FileError>> =>
    Promise.resolve({
      ok: false,
      error: new FileError(
        "not_supported",
        `${method} is not implemented for the Exo sandbox execution environment`,
      ),
    });

  return {
    cwd,

    // Purely syntactic: pi calls this to address a path without requiring it
    // to exist, so it must not touch the sandbox at all.
    async absolutePath(path) {
      return { ok: true, value: resolvePath(cwd, path) };
    },

    async joinPath(parts) {
      return { ok: true, value: resolvePath(cwd, parts.join("/")) };
    },

    async canonicalPath(path) {
      const target = resolvePath(cwd, path);
      // -m: resolve symlinks but do not require the final component to exist,
      // which is what pi expects when addressing a file about to be written.
      const outcome = await run(`readlink -m -- ${shellQuote(target)}`);
      if (outcome.exitCode !== 0) {
        return { ok: false, error: fileError(outcome, target) };
      }
      return { ok: true, value: outcome.stdout.trim() || target };
    },

    async exists(path) {
      const target = resolvePath(cwd, path);
      const outcome = await run(
        `test -e ${shellQuote(target)} && echo yes || echo no`,
      );
      if (outcome.exitCode !== 0) {
        return { ok: false, error: fileError(outcome, target) };
      }
      return { ok: true, value: outcome.stdout.trim() === "yes" };
    },

    async fileInfo(path) {
      const target = resolvePath(cwd, path);
      // %F kind, %s size, %Y mtime in seconds. -c is GNU stat; the sandbox
      // image is ubuntu, so this is the coreutils form rather than BSD's -f.
      const outcome = await run(`stat -c '%F|%s|%Y' -- ${shellQuote(target)}`);
      if (outcome.exitCode !== 0) {
        return { ok: false, error: fileError(outcome, target) };
      }
      const [kindText, sizeText, mtimeText] = outcome.stdout.trim().split("|");
      return {
        ok: true,
        value: {
          name: target.split("/").filter(Boolean).at(-1) ?? target,
          path: target,
          kind: statKindToFileKind(kindText),
          size: Number(sizeText) || 0,
          mtimeMs: (Number(mtimeText) || 0) * 1000,
        } satisfies FileInfo,
      };
    },

    async readTextFile(path) {
      const target = resolvePath(cwd, path);
      const bytes = await readBase64(run, target);
      if (!bytes.ok) {
        return bytes;
      }
      return { ok: true, value: bytes.value.toString("utf8") };
    },

    async readBinaryFile(path) {
      const target = resolvePath(cwd, path);
      const bytes = await readBase64(run, target);
      if (!bytes.ok) {
        return bytes;
      }
      return { ok: true, value: new Uint8Array(bytes.value) };
    },

    async writeFile(path, content) {
      const target = resolvePath(cwd, path);
      const buffer =
        typeof content === "string"
          ? Buffer.from(content, "utf8")
          : Buffer.from(content);
      const encoded = buffer.toString("base64");
      const directory = target.slice(0, target.lastIndexOf("/")) || "/";
      // Content never appears in the command text - only its base64 form
      // does - so no amount of quoting, newlines or heredoc markers inside
      // the file can break the command.
      const outcome = await run(
        `mkdir -p -- ${shellQuote(directory)} && ` +
          `printf %s ${shellQuote(encoded)} | base64 -d > ${shellQuote(target)}`,
      );
      if (outcome.exitCode !== 0) {
        return { ok: false, error: fileError(outcome, target) };
      }
      return { ok: true, value: undefined };
    },

    // Not called by the read/write/edit tools this env exists to serve.
    readTextLines: () => unsupported("readTextLines"),
    appendFile: () => unsupported("appendFile"),
    renameFile: () => unsupported("renameFile"),
    listDir: () => unsupported("listDir"),
    createDir: () => unsupported("createDir"),
    remove: () => unsupported("remove"),
    createTempDir: () => unsupported("createTempDir"),
    createTempFile: () => unsupported("createTempFile"),

    async exec(
      command: string,
      options?: ShellExecOptions,
    ): Promise<Result<ShellOutcome, ExecutionError>> {
      if (options?.abortSignal?.aborted) {
        return {
          ok: false,
          error: new ExecutionError("aborted", "command aborted before start"),
        };
      }
      try {
        const outcome = await run(command);
        return { ok: true, value: outcome };
      } catch (error) {
        return {
          ok: false,
          error: new ExecutionError(
            "spawn_error",
            error instanceof Error ? error.message : String(error),
          ),
        };
      }
    },

    async cleanup() {
      // The sandbox outlives any single turn and is torn down by Exo, not
      // here. Must not throw per the interface contract.
    },
  };
}

async function readBase64(
  run: (command: string) => Promise<ShellOutcome>,
  target: string,
): Promise<Result<Buffer, FileError>> {
  // pipefail matters: `tr` succeeds on empty input, so without it a missing
  // file reads back as an empty string instead of an error. That exact bug
  // corrupted a file in this repo once already (see file-tools.ts).
  const outcome = await run(
    `set -o pipefail && base64 < ${shellQuote(target)} | tr -d '\\n'`,
  );
  if (outcome.exitCode !== 0) {
    return { ok: false, error: fileError(outcome, target) };
  }
  return { ok: true, value: Buffer.from(outcome.stdout.trim(), "base64") };
}

async function runShell(
  context: TurnContext,
  command: string,
): Promise<ShellOutcome> {
  const result = await context.executeTool({
    functionName: "shell",
    arguments: { command } as JsonObject,
  });
  return normalizeShellResult(result);
}

// The shell result arrives in more than one shape depending on the call path
// (bare, or nested under value/details) - the same normalisation file-tools.ts
// and built-in-tools.ts each had to do.
function normalizeShellResult(result: ToolResult): ShellOutcome {
  const found = findShellRecord(result, 0);
  return {
    stdout: typeof found?.stdout === "string" ? found.stdout : "",
    stderr: typeof found?.stderr === "string" ? found.stderr : "",
    exitCode: typeof found?.exit_code === "number" ? found.exit_code : 0,
  };
}

function findShellRecord(
  node: unknown,
  depth: number,
): { stdout?: unknown; stderr?: unknown; exit_code?: unknown } | null {
  if (depth > 6 || node === null || typeof node !== "object") {
    return null;
  }
  const record = node as Record<string, unknown>;
  if (typeof record.stdout === "string" || typeof record.stderr === "string") {
    return record;
  }
  for (const value of Object.values(record)) {
    const found = findShellRecord(value, depth + 1);
    if (found) {
      return found;
    }
  }
  return null;
}

// Map a failed command to the closest FileErrorCode. Kept coarse on purpose:
// guessing a specific code from stderr text would be wrong more often than
// "unknown" is unhelpful.
function fileError(outcome: ShellOutcome, path: string): FileError {
  const stderr = outcome.stderr.toLowerCase();
  const code = stderr.includes("no such file")
    ? "not_found"
    : stderr.includes("permission denied")
      ? "permission_denied"
      : stderr.includes("is a directory")
        ? "is_directory"
        : stderr.includes("not a directory")
          ? "not_directory"
          : "unknown";
  return new FileError(
    code,
    outcome.stderr.trim() ||
      `command failed with exit code ${outcome.exitCode}`,
    path,
  );
}

function statKindToFileKind(kindText: string | undefined): FileInfo["kind"] {
  const text = (kindText ?? "").toLowerCase();
  if (text.includes("directory")) {
    return "directory";
  }
  if (text.includes("symbolic link")) {
    return "symlink";
  }
  return "file";
}

// Syntactic resolution only - no sandbox round-trip, because pi calls
// absolutePath on paths that do not exist yet.
function resolvePath(cwd: string, path: string): string {
  const combined = path.startsWith("/") ? path : `${cwd}/${path}`;
  const segments: string[] = [];
  for (const segment of combined.split("/")) {
    if (segment === "" || segment === ".") {
      continue;
    }
    if (segment === "..") {
      segments.pop();
      continue;
    }
    segments.push(segment);
  }
  return `/${segments.join("/")}`;
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}
