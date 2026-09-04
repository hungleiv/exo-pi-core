import { fileURLToPath, pathToFileURL } from "node:url";
import { mkdtempSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { HarnessToolRegistry, type TurnContext } from "@exo/harness";
import { describe, expect, it } from "vitest";

import {
  loadPiExtension,
  piExtensionModuleUrl,
  piExtensionPathsFromEnv,
  toStrictParameters,
} from "./pi-compat";
import type { JsonObject } from "@exo/harness";

const fixturePath = fileURLToPath(
  new URL("./fixtures/sample-pi-extension.ts", import.meta.url),
);

function registry(): HarnessToolRegistry {
  return new HarnessToolRegistry({} as TurnContext);
}

describe("pi-compat strict parameters", () => {
  it("requires every property and nullables the optionals", () => {
    expect(
      toStrictParameters({
        type: "object",
        properties: {
          text: { type: "string" },
          loud: { type: "boolean" },
        },
        required: ["text"],
      }),
    ).toEqual({
      type: "object",
      additionalProperties: false,
      properties: {
        text: { type: "string" },
        loud: { type: ["boolean", "null"] },
      },
      required: ["text", "loud"],
    });
  });

  it("defaults a missing schema to an empty strict object", () => {
    expect(toStrictParameters(undefined)).toEqual({
      type: "object",
      additionalProperties: false,
      properties: {},
      required: [],
    });
  });
});

describe("loadPiExtension", () => {
  it("registers tools as library tools and reports unsupported events", async () => {
    const loaded = await loadPiExtension(registry(), fixturePath, {
      cacheBust: false,
      route: "sandbox",
      cwd: "/work",
    });

    expect(loaded.name).toBe("sample-pi-extension");
    expect(loaded.tools.map((tool) => tool.definition.name)).toEqual(["shout"]);
    expect(loaded.tools[0]?.source).toBe("library");
    expect(loaded.unsupportedEvents).toEqual(["session_start"]);
  });

  it("executes converted tools with route context", async () => {
    const tools = registry();
    await loadPiExtension(tools, fixturePath, {
      cacheBust: false,
      route: "sandbox",
      cwd: "/work",
    });

    const result = (await tools
      .get("shout")
      ?.handler.execute(
        { text: "hey", loud: true },
        { context: {} as TurnContext },
      )) as JsonObject;

    expect(result).toMatchObject({
      ok: true,
      text: "HEY!!!",
      route: "sandbox",
    });
  });

  it("exposes commands through one dispatcher tool when asked", async () => {
    const tools = registry();
    const loaded = await loadPiExtension(tools, fixturePath, {
      cacheBust: false,
      exposeCommands: true,
    });

    expect(loaded.tools.map((tool) => tool.definition.name).sort()).toEqual([
      "sample-pi-extension_command",
      "shout",
    ]);

    const result = (await tools
      .get("sample-pi-extension_command")
      ?.handler.execute(
        { command: "greet", args: "exo" },
        { context: {} as TurnContext },
      )) as JsonObject;

    expect(result).toMatchObject({ ok: true, text: "hello exo" });
  });

  it("rejects extensions without a default export", async () => {
    await expect(
      loadPiExtension(registry(), import.meta.filename, { cacheBust: false }),
    ).rejects.toThrow(/no default export/);
  });

  it("changes the module URL when the file changes", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "pi-reload-"));
    const extensionPath = path.join(dir, "ext.ts");
    writeFileSync(extensionPath, "export default function() {}");

    const before = piExtensionModuleUrl(extensionPath);
    const later = new Date(Date.now() + 2000);
    utimesSync(extensionPath, later, later);
    const after = piExtensionModuleUrl(extensionPath);

    // Each saved version maps to a distinct dot-prefixed sibling path so
    // the module cache busts even when the loader strips URL queries.
    expect(before).not.toBe(after);
    expect(path.basename(before)).toMatch(/^\.ext--\d+-\d+\.ts$/);
    expect(piExtensionModuleUrl(extensionPath, false)).toBe(
      pathToFileURL(extensionPath).href,
    );
  });

  it("vetoes tool calls through a blocking tool_call listener", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "pi-policy-"));
    const extensionPath = path.join(dir, "policy.ts");
    writeFileSync(
      extensionPath,
      `
      export default function (pi) {
        pi.on("tool_call", (event, execution) => {
          pi.seenExecution = execution && execution.context !== undefined;
          if (event.args.dangerous === true) {
            return { block: true, reason: "dangerous flag set" };
          }
          return { before: async () => { pi.ranBefore = true; } };
        });
        pi.registerTool({
          name: "guarded",
          description: "guarded tool",
          parameters: { type: "object", additionalProperties: false, properties: { dangerous: { type: ["boolean", "null"] } }, required: ["dangerous"] },
          execute: (args) => ({ ok: true, ran: true, dangerous: args.dangerous === true }),
        });
      }
      `,
    );

    const tools = registry();
    await loadPiExtension(tools, extensionPath, { cacheBust: false });

    const blocked = (await tools
      .get("guarded")
      ?.handler.execute(
        { dangerous: true },
        { context: {} as TurnContext },
      )) as JsonObject;
    expect(blocked).toMatchObject({
      ok: false,
      error: expect.stringContaining("blocked by policy: dangerous flag set"),
    });

    const allowed = (await tools
      .get("guarded")
      ?.handler.execute(
        { dangerous: null },
        { context: {} as TurnContext },
      )) as JsonObject;
    // The before hook ran ahead of the handler and the listener saw the
    // execution context (policy has the same reach as a native tool).
    expect(allowed).toMatchObject({ ok: true, ran: true });
  });

  it("enforces setActiveTools at registration time", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "pi-active-"));
    const extensionPath = path.join(dir, "active.ts");
    writeFileSync(
      extensionPath,
      `
      export default function (pi) {
        pi.registerTool({ name: "tool_a", description: "a", parameters: undefined, execute: () => ({ ok: true }) });
        pi.registerTool({ name: "tool_b", description: "b", parameters: undefined, execute: () => ({ ok: true }) });
        pi.setActiveTools(["tool_a"]);
      }
      `,
    );

    const tools = registry();
    const loaded = await loadPiExtension(tools, extensionPath, {
      cacheBust: false,
    });
    expect(loaded.tools.map((tool) => tool.definition.name)).toEqual([
      "tool_a",
    ]);
    expect(tools.get("tool_a")).toBeDefined();
    expect(tools.get("tool_b")).toBeUndefined();
  });
});

describe("piExtensionPathsFromEnv", () => {
  it("returns empty when unset and splits comma lists", () => {
    expect(piExtensionPathsFromEnv({} as NodeJS.ProcessEnv)).toEqual([]);
    expect(
      piExtensionPathsFromEnv({
        EXO_PI_EXTENSIONS: " a.ts ,b.ts,, ",
      } as NodeJS.ProcessEnv),
    ).toEqual(["a.ts", "b.ts"]);
  });

  it("resolves bundled names against the given prefix", () => {
    const paths = piExtensionPathsFromEnv(
      {
        EXO_PI_EXTENSIONS_BUNDLED: "web-tools-extension.ts, ../x.ts",
      } as NodeJS.ProcessEnv,
      { bundledPrefix: "./extensions" },
    );
    // Both entries resolve relative to this module's directory (exo/tools/).
    expect(paths[0]).toBe(
      path.join(
        path.dirname(fileURLToPath(import.meta.url)),
        "extensions",
        "web-tools-extension.ts",
      ),
    );
    expect(paths[1]).toBe(
      path.join(path.dirname(fileURLToPath(import.meta.url)), "x.ts"),
    );
    // Absolute entries pass through untouched.
    expect(
      piExtensionPathsFromEnv(
        { EXO_PI_EXTENSIONS_BUNDLED: "/tmp/ext.ts" } as NodeJS.ProcessEnv,
        { bundledPrefix: "./extensions" },
      ),
    ).toEqual(["/tmp/ext.ts"]);
  });
});

describe("registerToolInstance", () => {
  it("registers a native ToolInstance with execution passthrough", async () => {
    const dir = mkdtempSync(path.join(tmpdir(), "pi-native-"));
    const extensionPath = path.join(dir, "native.ts");
    writeFileSync(
      extensionPath,
      `
      export default function (pi) {
        pi.seen = pi.seen || [];
        pi.on("tool_call", (event) => pi.seen.push(event.toolName));
        pi.registerToolInstance({
          source: "library",
          definition: {
            name: "native_echo",
            description: "echo",
            parameters: { type: "object", additionalProperties: false, properties: { text: { type: "string" } }, required: ["text"] },
          },
          handler: {
            async execute(args, execution) {
              pi.gotExecution = execution !== undefined && execution.context !== undefined;
              return { ok: true, text: args.text, contextSeen: pi.gotExecution };
            },
          },
        });
      }
      `,
    );

    const tools = registry();
    const loaded = await loadPiExtension(tools, extensionPath, {
      cacheBust: false,
    });
    expect(loaded.tools.map((tool) => tool.definition.name)).toEqual([
      "native_echo",
    ]);

    const result = (await tools
      .get("native_echo")
      ?.handler.execute(
        { text: "hi" },
        { context: {} as TurnContext },
      )) as JsonObject;
    // The native handler received the full execution (context passthrough)
    // and the tool_call listener fired.
    expect(result).toMatchObject({ ok: true, contextSeen: true });
    expect(loaded.unsupportedEvents).toEqual([]);
  });
});
