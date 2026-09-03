import { fileURLToPath } from "node:url";

import { HarnessToolRegistry, type TurnContext } from "@exo/harness";
import { describe, expect, it } from "vitest";

import { loadPiExtension, toStrictParameters } from "./pi-compat";
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
    await loadPiExtension(tools, fixturePath, { route: "sandbox", cwd: "/work" });

    const result = (await tools.get("shout")?.handler.execute(
      { text: "hey", loud: true },
      { context: {} as TurnContext },
    )) as JsonObject;

    expect(result).toMatchObject({ ok: true, text: "HEY!!!", route: "sandbox" });
  });

  it("exposes commands through one dispatcher tool when asked", async () => {
    const tools = registry();
    const loaded = await loadPiExtension(tools, fixturePath, { exposeCommands: true });

    expect(loaded.tools.map((tool) => tool.definition.name).sort()).toEqual([
      "sample-pi-extension_command",
      "shout",
    ]);

    const result = (await tools
      .get("sample-pi-extension_command")
      ?.handler.execute({ command: "greet", args: "exo" }, { context: {} as TurnContext })) as JsonObject;

    expect(result).toMatchObject({ ok: true, text: "hello exo" });
  });

  it("rejects extensions without a default export", async () => {
    await expect(loadPiExtension(registry(), import.meta.filename)).rejects.toThrow(
      /no default export/,
    );
  });
});
