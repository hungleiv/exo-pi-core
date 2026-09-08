import { describe, expect, it } from "vitest";

// Mirrors DIFF_MAX_LINES in file-tools.ts.
const DIFF_LINE_BUDGET = 41;

// Linux's MAX_ARG_STRLEN: 32 pages, the per-argument cap that a single
// `bash -lc <script>` call is measured against.
const FAKE_MAX_ARG_STRLEN = 131072;

import type {
  JsonObject,
  ToolExecutionContext,
  ToolRequest,
} from "@exo/harness";

import {
  editToolInstance,
  readToolInstance,
  writeToolInstance,
} from "./file-tools";

// The only seam these tools have into the sandbox filesystem is
// execution.context.executeTool({functionName: "shell"}) - see file-tools.ts's
// header. Faking just that call, the way memory-tools.test.ts fakes just the
// artifact methods it needs, exercises the real base64/pipefail/quoting logic
// without a live sandbox.
function fakeShellExecution(
  respond: (command: string) => {
    exit_code: number;
    stdout: string;
    stderr: string;
  },
): ToolExecutionContext {
  return {
    context: {
      async executeTool(request: ToolRequest) {
        if (request.functionName !== "shell") {
          throw new Error(`unexpected tool call: ${request.functionName}`);
        }
        const command = (request.arguments as { command?: unknown }).command;
        if (typeof command !== "string") {
          throw new Error("shell call missing command");
        }
        return respond(command);
      },
      // Nothing under test reads other TurnContext fields.
    } as unknown as ToolExecutionContext["context"],
  };
}

// Minimal fake of the actual sandbox commands these tools shell out to, so
// tests don't need a real container: write pipes base64 through `base64 -d`,
// read pipes a file through `base64`. Good enough to exercise the tools'
// own logic (encoding, error propagation, truncation) rather than bash's.
// `files` holds text, encoded as UTF-8 on the way out, which is what almost
// every test wants. `binary` is a separate channel for files whose bytes are
// not text at all - an image's 0x89 byte does not survive a round trip through
// a UTF-8 string, so storing one in `files` would silently corrupt it and the
// test would be asserting against the corruption.
function fakeSandboxFs() {
  const files = new Map<string, string>();
  const binary = new Map<string, Buffer>();
  // Partially-written base64, visible only to the chunked write path. A test
  // asserting on `files` therefore cannot see a half-finished write, which is
  // the property the staging file exists to provide.
  const staging = new Map<string, string>();
  const commandLengths: number[] = [];
  const execution = fakeShellExecution((command) => {
    // The sandbox seam is one `bash -lc <script>` call, and Linux caps a
    // single argv element at MAX_ARG_STRLEN (32 pages). Enforcing it here is
    // what makes the large-write tests real: without it the fake happily
    // accepts a 200KB command that the kernel rejects with E2BIG, and the
    // tests pass against a bug that is live in production.
    if (command.length > FAKE_MAX_ARG_STRLEN) {
      throw new Error(
        `failed to start sandbox command: /bin/bash -lc ...\ncaused by: Argument list too long (os error 7)`,
      );
    }
    const writeMatch = command.match(
      /printf %s '((?:[^'\\]|\\.)*)' \| base64 -d > '((?:[^'\\]|\\.)*)'/,
    );
    if (writeMatch) {
      const [, encoded, path] = writeMatch;
      files.set(
        fakePath(unquote(path)),
        Buffer.from(unquote(encoded), "base64").toString("utf8"),
      );
      return { exit_code: 0, stdout: "", stderr: "" };
    }
    const readMatch = command.match(
      /set -o pipefail && base64 < '((?:[^'\\]|\\.)*)' \| tr -d '\\n'/,
    );
    if (readMatch) {
      const path = fakePath(unquote(readMatch[1]));
      const rawBytes = binary.get(path);
      if (rawBytes) {
        return {
          exit_code: 0,
          stdout: rawBytes.toString("base64"),
          stderr: "",
        };
      }
      if (!files.has(path)) {
        // Real base64 exits non-zero on a missing file; pipefail (added after
        // the incident this file's header describes) is what makes that
        // reach here instead of being swallowed by tr's own success.
        return {
          exit_code: 1,
          stdout: "",
          stderr: `base64: ${path}: No such file or directory`,
        };
      }
      return {
        exit_code: 0,
        stdout: Buffer.from(files.get(path)!, "utf8").toString("base64"),
        stderr: "",
      };
    }
    // --- chunked write path -------------------------------------------
    // Modelled closely enough to catch a real ordering mistake: the staging
    // file accumulates, and only the final mv makes the new content visible
    // at the target path.
    const chunkMatch = command.match(
      /^printf %s '((?:[^'\\]|\\.)*)' (>>?) '((?:[^'\\]|\\.)*)'$/,
    );
    if (chunkMatch) {
      const [, encoded, redirect, path] = chunkMatch;
      const key = fakePath(unquote(path));
      const chunk = unquote(encoded);
      staging.set(
        key,
        redirect === ">" ? chunk : (staging.get(key) ?? "") + chunk,
      );
      commandLengths.push(command.length);
      return { exit_code: 0, stdout: "", stderr: "" };
    }
    const finalizeMatch = command.match(
      /^set -o pipefail && base64 -d < '((?:[^'\\]|\\.)*)' > '((?:[^'\\]|\\.)*)' && mv -- '((?:[^'\\]|\\.)*)' '((?:[^'\\]|\\.)*)' && rm -f '((?:[^'\\]|\\.)*)'$/,
    );
    if (finalizeMatch) {
      const stagingKey = fakePath(unquote(finalizeMatch[1]));
      const target = fakePath(unquote(finalizeMatch[4]));
      const accumulated = staging.get(stagingKey);
      if (accumulated === undefined) {
        return {
          exit_code: 1,
          stdout: "",
          stderr: `base64: ${stagingKey}: No such file or directory`,
        };
      }
      files.set(target, Buffer.from(accumulated, "base64").toString("utf8"));
      staging.delete(stagingKey);
      return { exit_code: 0, stdout: "", stderr: "" };
    }
    if (/^mkdir -p -- "\$\(dirname -- '.*'\)"$/.test(command)) {
      return { exit_code: 0, stdout: "", stderr: "" };
    }
    if (command.startsWith("rm -f ")) {
      return { exit_code: 0, stdout: "", stderr: "" };
    }
    throw new Error(`unrecognized command in fake sandbox: ${command}`);
  });
  return { execution, files, binary, staging, commandLengths };
}

function unquote(shellSingleQuoted: string): string {
  return shellSingleQuoted.replaceAll("'\\''", "'");
}

// A Map keyed by the literal path string would treat /a//b and /a/./b as two
// different files; a real filesystem does not. Collapsing them here keeps the
// fake honest about which spellings address one file.
function fakePath(path: string): string {
  const segments = path.split("/").filter((s) => s !== "" && s !== ".");
  return `${path.startsWith("/") ? "/" : ""}${segments.join("/")}`;
}

describe("write", () => {
  it("writes file content and reports bytes written", async () => {
    const { execution, files } = fakeSandboxFs();
    const result = await writeToolInstance().handler.execute(
      { path: "/tmp/bx/greeting.txt", content: "hello world" },
      execution,
    );

    expect(files.get("/tmp/bx/greeting.txt")).toBe("hello world");
    expect(result).toEqual({
      path: "/tmp/bx/greeting.txt",
      bytes_written: 11,
    });
  });

  it("round-trips content containing quotes and newlines with no escaping from the caller", async () => {
    const { execution, files } = fakeSandboxFs();
    const content = 'line one\nit\'s a "test"\nline three';
    await writeToolInstance().handler.execute(
      { path: "/tmp/bx/tricky.txt", content },
      execution,
    );

    expect(files.get("/tmp/bx/tricky.txt")).toBe(content);
  });

  // Paging is the recovery path for a cut-off read, and for the part of a
  // large shell result that got spilled to a file. Before this, a truncated
  // read had no way forward except re-reading the file whole - which returns
  // the same first 2,000 lines again - so a model had to fall back to
  // `shell tail/sed` instead.
  it("returns a window and says where to continue from", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set(
      "/tmp/bx/lines.txt",
      Array.from({ length: 10 }, (_, i) => `line${i + 1}`).join("\n"),
    );

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/lines.txt", offset: 3, limit: 2 },
      execution,
    )) as Record<string, unknown>;

    expect(result.content).toBe("line3\nline4");
    expect(result.truncated).toBe(true);
    expect(result.remaining_lines).toBe(6);
    expect(result.next_offset).toBe(5);
  });

  it("reads to the end without reporting a continuation", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/lines.txt", "a\nb\nc");

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/lines.txt", offset: 2, limit: 5 },
      execution,
    )) as Record<string, unknown>;

    expect(result.content).toBe("b\nc");
    expect(result.truncated).toBe(false);
    // Absent rather than zero: nothing left to read at all.
    expect("next_offset" in result).toBe(false);
  });

  // Strict mode has no optional properties, so the model is forced to send
  // every key and says "unset" by sending null. Treating that as an error
  // would make the common case fail.
  it("treats null offset and limit as unset", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/lines.txt", "a\nb");

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/lines.txt", offset: null, limit: null },
      execution,
    )) as Record<string, unknown>;

    expect(result.content).toBe("a\nb");
    expect(result.truncated).toBe(false);
  });

  it("rejects a nonsense offset instead of silently reading from the start", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/lines.txt", "a\nb");

    await expect(
      readToolInstance().handler.execute(
        { path: "/tmp/bx/lines.txt", offset: 0, limit: null },
        execution,
      ),
    ).rejects.toThrow(/offset must be a positive integer or null/);
  });

  it("still reports a continuation when the size cap cuts the window short", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set(
      "/tmp/bx/huge.txt",
      Array.from({ length: 3_000 }, (_, i) => `line${i + 1}`).join("\n"),
    );

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/huge.txt", offset: null, limit: null },
      execution,
    )) as Record<string, unknown>;

    expect(result.truncated).toBe(true);
    expect(result.next_offset).toBe(2_001);
    expect(result.remaining_lines).toBe(1_000);
  });

  it("rejects arguments outside the declared schema", async () => {
    const { execution } = fakeSandboxFs();
    await expect(
      writeToolInstance().handler.execute(
        { path: "/tmp/x", content: "y", extra: "z" } as unknown as JsonObject,
        execution,
      ),
    ).rejects.toThrow("tool argument extra is not allowed");
  });

  it("surfaces a non-zero shell exit as a thrown error instead of succeeding silently", async () => {
    const execution = fakeShellExecution(() => ({
      exit_code: 1,
      stdout: "",
      stderr: "permission denied",
    }));
    await expect(
      writeToolInstance().handler.execute(
        { path: "/root/blocked.txt", content: "x" },
        execution,
      ),
    ).rejects.toThrow(
      /write failed for \/root\/blocked.txt.*permission denied/,
    );
  });
});

// The t8-fuzzyedit failure: a model targets a phrase containing curly
// punctuation, retypes it with the ASCII lookalikes its own output
// naturally produces, and exact matching rejects the whole call. Measured
// 0/5 before this; pi's own edit recovered every time.
it("matches through curly quotes, apostrophes and dashes", async () => {
  const { execution, files } = fakeSandboxFs();
  files.set("/tmp/bx/note.txt", "The motto is: “Don’t stop” — keep going.");

  const result = (await editToolInstance().handler.execute(
    {
      path: "/tmp/bx/note.txt",
      edits: [{ old_text: '"Don\'t stop" - keep', new_text: "ONWARD keep" }],
    },
    execution,
  )) as Record<string, unknown>;

  expect(result.edits_applied).toBe(1);
  // Reported, not silent: the file differs from what was literally asked for.
  expect(result.fuzzy_matches).toBe(1);
  expect(files.get("/tmp/bx/note.txt")).toBe(
    "The motto is: ONWARD keep going.",
  );
});

// Only the matched span is rewritten - normalization must not leak into
// the rest of the file and quietly flatten punctuation the edit never
// targeted.
it("leaves typography outside the edited span untouched", async () => {
  const { execution, files } = fakeSandboxFs();
  files.set("/tmp/bx/note.txt", "keep “this” — replace “that” — end");

  await editToolInstance().handler.execute(
    {
      path: "/tmp/bx/note.txt",
      edits: [{ old_text: 'replace "that"', new_text: "REPLACED" }],
    },
    execution,
  );

  expect(files.get("/tmp/bx/note.txt")).toBe("keep “this” — REPLACED — end");
});

it("reports no fuzzy match when the text was found literally", async () => {
  const { execution, files } = fakeSandboxFs();
  files.set("/tmp/bx/note.txt", "plain ascii text here");

  const result = (await editToolInstance().handler.execute(
    {
      path: "/tmp/bx/note.txt",
      edits: [{ old_text: "ascii", new_text: "utf8" }],
    },
    execution,
  )) as Record<string, unknown>;

  expect("fuzzy_matches" in result).toBe(false);
});

// Normalizing must never turn an ambiguous target into an accepted one:
// guessing which of two identical spots was meant is exactly the mistake
// the exact-match rule exists to prevent.
it("still rejects a target that normalizes to more than one place", async () => {
  const { execution, files } = fakeSandboxFs();
  files.set("/tmp/bx/note.txt", "say “hi” then say ‘hi’ again");

  await expect(
    editToolInstance().handler.execute(
      {
        path: "/tmp/bx/note.txt",
        edits: [{ old_text: "hi", new_text: "bye" }],
      },
      execution,
    ),
  ).rejects.toThrow(/must match exactly once/);
  expect(files.get("/tmp/bx/note.txt")).toBe("say “hi” then say ‘hi’ again");
});

describe("read", () => {
  it("reads back exactly what was written", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/greeting.txt", "hello world");

    const result = await readToolInstance().handler.execute(
      { path: "/tmp/bx/greeting.txt" },
      execution,
    );

    expect(result).toEqual({
      path: "/tmp/bx/greeting.txt",
      content: "hello world",
      truncated: false,
    });
  });

  // The bug this guards: without `set -o pipefail`, a failing `base64` piped
  // into `tr` (which succeeds even on empty input) reported exit 0 with empty
  // stdout - a missing file read back as "", not an error. See
  // file-tools.ts's readFile comment and the Case 1 RSI incident it
  // references, where that false-empty read led the model to overwrite a
  // non-empty file it believed was blank.
  it("raises an error for a missing file rather than returning empty content", async () => {
    const { execution } = fakeSandboxFs();
    await expect(
      readToolInstance().handler.execute(
        { path: "/tmp/bx/missing.txt" },
        execution,
      ),
    ).rejects.toThrow(/read failed for \/tmp\/bx\/missing.txt/);
  });

  it("marks long content as truncated", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/big.txt", "x".repeat(60 * 1024));

    const result = await readToolInstance().handler.execute(
      { path: "/tmp/bx/big.txt" },
      execution,
    );

    expect(result).toMatchObject({ truncated: true });
    expect((result as { content: string }).content.length).toBeLessThan(
      60 * 1024,
    );
  });

  it("rejects arguments outside the declared schema", async () => {
    const { execution } = fakeSandboxFs();
    await expect(
      readToolInstance().handler.execute(
        { path: "/tmp/x", verbose: true } as unknown as JsonObject,
        execution,
      ),
    ).rejects.toThrow("tool argument verbose is not allowed");
  });
});

describe("edit", () => {
  it("applies a unique replacement", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/greeting.txt", "hello world");

    const result = await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/greeting.txt",
        edits: [{ old_text: "world", new_text: "there" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/greeting.txt")).toBe("hello there");
    expect(result).toEqual({
      path: "/tmp/bx/greeting.txt",
      edits_applied: 1,
      first_changed_line: 1,
      diff: "- 1 hello world\n+ 1 hello there",
    });
  });

  // A model that only sees "edits_applied: 1" cannot tell an edit that
  // landed where it meant from one that did not - and after fuzzy matching
  // the bytes written may differ from the old_text it typed. The diff is
  // what makes that visible without a follow-up read.
  it("reports the changed region with line numbers and context", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set(
      "/tmp/bx/poem.txt",
      ["alpha", "beta", "gamma", "delta", "epsilon"].join("\n"),
    );

    const result = (await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/poem.txt",
        edits: [{ old_text: "gamma", new_text: "GAMMA" }],
      },
      execution,
    )) as Record<string, unknown>;

    expect(result.first_changed_line).toBe(3);
    expect(result.diff).toBe(
      [
        "  1 alpha",
        "  2 beta",
        "- 3 gamma",
        "+ 3 GAMMA",
        "  4 delta",
        "  5 epsilon",
      ].join("\n"),
    );
  });

  // The result travels through compactToolResultForModel, which cuts a tool
  // result to ~4,000 characters. A whole-file diff would be truncated into
  // uselessness, so a large rewrite has to report a bounded excerpt instead.
  // A single before/after comparison reports everything between two distant
  // changes as one region: measured, edits at lines 3 and 200 of a 300-line
  // file gave a 401-line "diff" that the cap trimmed to 40 lines of
  // unchanged context, with the second change never visible at all.
  it("shows both changes when edits are far apart", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set(
      "/tmp/bx/far.txt",
      Array.from({ length: 300 }, (_, i) => `line-${i + 1}`).join("\n"),
    );

    const result = (await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/far.txt",
        edits: [
          { old_text: "line-3\nline-4", new_text: "CHANGED-3\nline-4" },
          { old_text: "line-200\nline-201", new_text: "CHANGED-200\nline-201" },
        ],
      },
      execution,
    )) as Record<string, unknown>;

    const diff = String(result.diff);
    expect(result.first_changed_line).toBe(3);
    // Both edits visible, and the untouched middle is not spelled out.
    expect(diff).toContain("+ 3 CHANGED-3");
    expect(diff).toContain("+ 200 CHANGED-200");
    expect(diff).not.toContain("line-100");
    expect(diff.split("\n").length).toBeLessThanOrEqual(DIFF_LINE_BUDGET);
  });

  it("caps the diff instead of dumping a whole rewritten file", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set(
      "/tmp/bx/big.txt",
      Array.from({ length: 500 }, (_, i) => `old-line-${i}`).join("\n"),
    );

    const result = (await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/big.txt",
        edits: [
          {
            old_text: Array.from(
              { length: 500 },
              (_, i) => `old-line-${i}`,
            ).join("\n"),
            new_text: Array.from(
              { length: 500 },
              (_, i) => `new-line-${i}`,
            ).join("\n"),
          },
        ],
      },
      execution,
    )) as Record<string, unknown>;

    const diffLines = String(result.diff).split("\n");
    expect(diffLines.length).toBeLessThanOrEqual(41);
    expect(diffLines.at(-1)).toMatch(/more diff line\(s\) not shown/);
  });

  it("omits diff fields when an edit changes nothing", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/same.txt", "unchanged");

    const result = (await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/same.txt",
        edits: [{ old_text: "unchanged", new_text: "unchanged" }],
      },
      execution,
    )) as Record<string, unknown>;

    expect("diff" in result).toBe(false);
    expect("first_changed_line" in result).toBe(false);
  });

  it("applies multiple edits in order", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/a.txt", "one two three");

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/a.txt",
        edits: [
          { old_text: "one", new_text: "1" },
          { old_text: "three", new_text: "3" },
        ],
      },
      execution,
    );

    expect(files.get("/tmp/bx/a.txt")).toBe("1 two 3");
  });

  it("rejects and leaves the file untouched when old_text is not unique", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/dup.txt", "foo foo");

    await expect(
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/dup.txt",
          edits: [{ old_text: "foo", new_text: "bar" }],
        },
        execution,
      ),
    ).rejects.toThrow(/matches 2 times/);
    expect(files.get("/tmp/bx/dup.txt")).toBe("foo foo");
  });

  it("rejects and leaves the file untouched when old_text is absent", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/a.txt", "hello");

    await expect(
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/a.txt",
          edits: [{ old_text: "missing", new_text: "x" }],
        },
        execution,
      ),
    ).rejects.toThrow(/matches 0 times/);
    expect(files.get("/tmp/bx/a.txt")).toBe("hello");
  });

  it("rejects arguments outside the declared schema", async () => {
    const { execution } = fakeSandboxFs();
    await expect(
      editToolInstance().handler.execute(
        {
          path: "/tmp/x",
          edits: [{ old_text: "a", new_text: "b" }],
          dryRun: true,
        } as unknown as JsonObject,
        execution,
      ),
    ).rejects.toThrow("tool argument dryRun is not allowed");
  });
});

describe("line endings and BOM", () => {
  it("edits a CRLF file using old_text written with plain newlines", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/crlf.txt", "alpha\r\nbeta\r\ngamma\r\n");

    // The model reads text, not bytes, so it writes \n even though the file
    // on disk is \r\n. Before normalization this matched zero times and the
    // edit was rejected outright.
    const result = await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/crlf.txt",
        edits: [{ old_text: "beta\ngamma", new_text: "beta\ndelta" }],
      },
      execution,
    );

    expect((result as { edits_applied: number }).edits_applied).toBe(1);
    // Rewritten with the endings it arrived with, not silently converted.
    expect(files.get("/tmp/bx/crlf.txt")).toBe("alpha\r\nbeta\r\ndelta\r\n");
  });

  it("keeps a file's CRLF endings when new_text spans lines", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/crlf2.txt", "one\r\ntwo\r\n");

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/crlf2.txt",
        edits: [{ old_text: "two", new_text: "two\nthree" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/crlf2.txt")).toBe("one\r\ntwo\r\nthree\r\n");
  });

  // A BOM needs no special handling here - see the note above
  // detectLineEnding in file-tools.ts - but it must survive an edit, which is
  // what this pins.
  it("leaves a BOM in place across an edit", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/bom.txt", "﻿header\nbody\n");

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/bom.txt",
        edits: [{ old_text: "header", new_text: "title" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/bom.txt")).toBe("﻿title\nbody\n");
  });

  it("leaves an LF file on LF", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/lf.txt", "one\ntwo\n");

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/lf.txt",
        edits: [{ old_text: "two", new_text: "three" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/lf.txt")).toBe("one\nthree\n");
  });
});

describe("concurrent mutation of one file", () => {
  // harness-pi-core.ts currently sets toolExecution "sequential", so these
  // overlaps cannot arise in production today - these tests exercise the
  // per-file lock directly, by calling the tools concurrently the way a
  // parallel loop would. See the note above withFileLock for why the lock
  // exists anyway: it is what would let that harness-wide serialization be
  // reconsidered on latency grounds instead of correctness ones.
  it("applies both edits when two edit calls target the same file", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/shared.txt", "alpha\nbeta\n");

    await Promise.all([
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/shared.txt",
          edits: [{ old_text: "alpha", new_text: "ALPHA" }],
        },
        execution,
      ),
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/shared.txt",
          edits: [{ old_text: "beta", new_text: "BETA" }],
        },
        execution,
      ),
    ]);

    expect(files.get("/tmp/bx/shared.txt")).toBe("ALPHA\nBETA\n");
  });

  it("does not serialize edits to different files", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/one.txt", "x");
    files.set("/tmp/bx/two.txt", "x");

    await Promise.all([
      editToolInstance().handler.execute(
        { path: "/tmp/bx/one.txt", edits: [{ old_text: "x", new_text: "1" }] },
        execution,
      ),
      editToolInstance().handler.execute(
        { path: "/tmp/bx/two.txt", edits: [{ old_text: "x", new_text: "2" }] },
        execution,
      ),
    ]);

    expect(files.get("/tmp/bx/one.txt")).toBe("1");
    expect(files.get("/tmp/bx/two.txt")).toBe("2");
  });

  it("treats /a//b and /a/./b as the same file", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/alias.txt", "alpha\nbeta\n");

    await Promise.all([
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx//alias.txt",
          edits: [{ old_text: "alpha", new_text: "ALPHA" }],
        },
        execution,
      ),
      editToolInstance().handler.execute(
        {
          path: "/tmp/./bx/alias.txt",
          edits: [{ old_text: "beta", new_text: "BETA" }],
        },
        execution,
      ),
    ]);

    expect(files.get("/tmp/bx/alias.txt")).toBe("ALPHA\nBETA\n");
  });

  it("releases the lock when an edit fails, so a later edit still runs", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/after.txt", "alpha\n");

    const results = await Promise.allSettled([
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/after.txt",
          edits: [{ old_text: "missing", new_text: "x" }],
        },
        execution,
      ),
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/after.txt",
          edits: [{ old_text: "alpha", new_text: "ALPHA" }],
        },
        execution,
      ),
    ]);

    expect(results[0].status).toBe("rejected");
    expect(results[1].status).toBe("fulfilled");
    expect(files.get("/tmp/bx/after.txt")).toBe("ALPHA\n");
  });

  it("serializes a write against an edit on the same file", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/mixed.txt", "alpha\n");

    await Promise.all([
      editToolInstance().handler.execute(
        {
          path: "/tmp/bx/mixed.txt",
          edits: [{ old_text: "alpha", new_text: "ALPHA" }],
        },
        execution,
      ),
      writeToolInstance().handler.execute(
        { path: "/tmp/bx/mixed.txt", content: "replaced\n" },
        execution,
      ),
    ]);

    // The write is last in the queue, so it wins - but it ran after the edit
    // completed rather than on top of a half-finished read-modify-write.
    expect(files.get("/tmp/bx/mixed.txt")).toBe("replaced\n");
  });
});

describe("mixed line endings", () => {
  // pi would rewrite every line of this file to the ending of its first line.
  // Leaving untouched lines byte-identical matters more than uniformity.
  it("does not convert lines the edit never touched", async () => {
    const { execution, files } = fakeSandboxFs();
    const mixed = "alpha\r\nbeta\ngamma\r\n";
    files.set("/tmp/bx/mixed-eol.txt", mixed);

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/mixed-eol.txt",
        edits: [{ old_text: "gamma", new_text: "delta" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/mixed-eol.txt")).toBe("alpha\r\nbeta\ndelta\r\n");
  });

  it("still matches exact CRLF old_text in a mixed file", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/mixed2.txt", "alpha\r\nbeta\ngamma\r\n");

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/mixed2.txt",
        edits: [{ old_text: "alpha\r\nbeta", new_text: "ALPHA\r\nBETA" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/mixed2.txt")).toBe("ALPHA\r\nBETA\ngamma\r\n");
  });
});

describe("read of an image file", () => {
  // A 1x1 PNG. Decoding these bytes as UTF-8 - which is what read did before -
  // yields replacement-character noise that costs context and tells the model
  // nothing, with no argument it could retry with to get the real thing.
  const PNG_1X1 = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    "base64",
  );

  it("returns the image as base64 with its media type, not as text", async () => {
    const { execution, binary } = fakeSandboxFs();
    binary.set("/tmp/bx/dot.png", PNG_1X1);

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/dot.png", offset: null, limit: null },
      execution,
    )) as Record<string, unknown>;

    expect(result.media_type).toBe("image/png");
    expect(result.bytes).toBe(PNG_1X1.length);
    expect(result.image_base64).toBe(PNG_1X1.toString("base64"));
    // The text-shaped fields must be absent: an image has no lines to page.
    expect("content" in result).toBe(false);
    expect("next_offset" in result).toBe(false);
  });

  it("detects jpeg, gif and webp by signature", async () => {
    const cases: [string, Buffer, string][] = [
      [
        "/tmp/bx/a.jpg",
        Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00]),
        "image/jpeg",
      ],
      ["/tmp/bx/a.gif", Buffer.from("GIF89a-rest"), "image/gif"],
      [
        "/tmp/bx/a.webp",
        Buffer.concat([
          Buffer.from("RIFF"),
          Buffer.from([0, 0, 0, 0]),
          Buffer.from("WEBPmore"),
        ]),
        "image/webp",
      ],
    ];
    for (const [path, bytes, expected] of cases) {
      const { execution, binary } = fakeSandboxFs();
      binary.set(path, bytes);
      const result = (await readToolInstance().handler.execute(
        { path, offset: null, limit: null },
        execution,
      )) as Record<string, unknown>;
      expect(result.media_type).toBe(expected);
    }
  });

  it("still reads a text file that merely mentions PNG as text", async () => {
    const { execution, files } = fakeSandboxFs();
    files.set("/tmp/bx/notes.txt", "PNG is a format\nGIF is another");

    const result = (await readToolInstance().handler.execute(
      { path: "/tmp/bx/notes.txt", offset: null, limit: null },
      execution,
    )) as Record<string, unknown>;

    expect(result.content).toBe("PNG is a format\nGIF is another");
    expect("media_type" in result).toBe(false);
  });

  // The sandbox has a shell, so the model can act on this rather than being
  // told only that it failed.
  it("refuses an oversized image with a recoverable instruction", async () => {
    const { execution, binary } = fakeSandboxFs();
    const huge = Buffer.concat([
      Buffer.from([0xff, 0xd8, 0xff]),
      Buffer.alloc(4 * 1024 * 1024),
    ]);
    binary.set("/tmp/bx/huge.jpg", huge);

    await expect(
      readToolInstance().handler.execute(
        { path: "/tmp/bx/huge.jpg", offset: null, limit: null },
        execution,
      ),
    ).rejects.toThrow(/resize it first/);
  });
});

describe("large file writes", () => {
  // Linux caps one argv element at MAX_ARG_STRLEN (131072 bytes), and the
  // whole shell script is one such element. Found live: an agent editing a
  // 148KB source file got "Argument list too long (os error 7)" and lost the
  // edit. Anything from ~96KB of content upward was unwritable.
  function bigText(bytes: number): string {
    // Repeating but not uniform, so a chunk boundary landing in the wrong
    // place corrupts the content visibly rather than by luck looking right.
    let out = "";
    for (let i = 0; out.length < bytes; i += 1) {
      out += `line ${i} ${"x".repeat(50)}\n`;
    }
    return out.slice(0, bytes);
  }

  it("writes a file far larger than the single-argument limit", async () => {
    const { execution, files } = fakeSandboxFs();
    const content = bigText(400_000);

    await writeToolInstance().handler.execute(
      { path: "/tmp/bx/big.txt", content },
      execution,
    );

    expect(files.get("/tmp/bx/big.txt")).toBe(content);
  });

  it("keeps every shell command under the argv limit", async () => {
    const { execution, commandLengths } = fakeSandboxFs();

    await writeToolInstance().handler.execute(
      { path: "/tmp/bx/big.txt", content: bigText(400_000) },
      execution,
    );

    expect(commandLengths.length).toBeGreaterThan(1);
    for (const length of commandLengths) {
      expect(length).toBeLessThan(FAKE_MAX_ARG_STRLEN);
    }
  });

  it("still uses a single command for an ordinary small file", async () => {
    const { execution, commandLengths, files } = fakeSandboxFs();

    await writeToolInstance().handler.execute(
      { path: "/tmp/bx/small.txt", content: "hello" },
      execution,
    );

    // The chunked path never ran, so it recorded nothing.
    expect(commandLengths).toEqual([]);
    expect(files.get("/tmp/bx/small.txt")).toBe("hello");
  });

  it("leaves the previous content in place until the write completes", async () => {
    const { execution, files, staging } = fakeSandboxFs();
    files.set("/tmp/bx/big.txt", "original content");
    const content = bigText(200_000);

    await writeToolInstance().handler.execute(
      { path: "/tmp/bx/big.txt", content },
      execution,
    );

    expect(files.get("/tmp/bx/big.txt")).toBe(content);
    // Staging cleaned up rather than left behind next to the real file.
    expect([...staging.keys()]).toEqual([]);
  });

  it("edits a large existing file end to end", async () => {
    const { execution, files } = fakeSandboxFs();
    const content = `${bigText(200_000)}\nNEEDLE marker line\n`;
    files.set("/tmp/bx/big.ts", content);

    await editToolInstance().handler.execute(
      {
        path: "/tmp/bx/big.ts",
        edits: [{ old_text: "NEEDLE marker line", new_text: "REPLACED line" }],
      },
      execution,
    );

    expect(files.get("/tmp/bx/big.ts")).toContain("REPLACED line");
    expect(files.get("/tmp/bx/big.ts")).not.toContain("NEEDLE");
  });
});
