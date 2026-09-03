import { fileURLToPath } from "node:url";
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

    expect(before).not.toBe(after);
    expect(piExtensionModuleUrl(extensionPath, false)).not.toContain("?v=");
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
});
