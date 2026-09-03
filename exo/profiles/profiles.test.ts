import { HarnessToolRegistry, type TurnContext } from "@exo/harness";
import { describe, expect, it } from "vitest";

import { resolveExoProfile } from "./index";

describe("Exo profiles", () => {
  it("defaults to the practical profile", () => {
    expect(resolveExoProfile(undefined).name).toBe("practical");
  });

  it("gives bootstrap exactly the recovery capabilities", () => {
    const profile = resolveExoProfile("bootstrap");
    const context = {
      agentConfig: { enableAgentToolCreation: false },
    } as TurnContext;
    const builtInNames = profile.builtInToolNames(context);
    expect(builtInNames).toEqual(["shell", "inspect_tools", "manage_tool"]);

    const tools = new HarnessToolRegistry(context);
    profile.registerTools(tools, context);
    expect([
      ...builtInNames,
      ...tools.definitions().map(({ name }) => name),
    ]).toEqual([
      "shell",
      "inspect_tools",
      "manage_tool",
      "rebuild_and_restart_exo",
    ]);
  });

  it("keeps practical extensions classified as library tools", async () => {
    const profile = resolveExoProfile("practical");
    const context = {
      agentConfig: { enableAgentToolCreation: false },
    } as TurnContext;
    const tools = new HarnessToolRegistry(context);
    expect(profile.builtInToolNames(context)).toEqual([
      "shell",
      "inspect_tools",
      "manage_tool",
    ]);
    await profile.registerTools(tools, context);

    // Core evolution tools stay profile-owned (host tools report source
    // built_in); the optional toolsets load as pi extensions
    // (EXO_PI_EXTENSIONS_BUNDLED in real runs, unset here).
    expect(tools.get("snapshot_sandbox")?.source).toBe("built_in");
    expect(tools.get("rebuild_and_restart_exo")?.source).toBe("built_in");
    expect(tools.get("web_search")).toBeUndefined();
    expect(tools.get("create_adapter")).toBeUndefined();
    expect([
      ...profile.builtInToolNames(context),
      ...tools
        .instances()
        .filter(({ source }) => source === "built_in")
        .map(({ definition }) => definition.name),
    ]).toEqual([
      "shell",
      "inspect_tools",
      "manage_tool",
      "get_sandbox_status",
      "list_sandbox_snapshots",
      "snapshot_sandbox",
      "rewind_sandbox",
      "rebuild_and_restart_exo",
    ]);
  });

  it("loads bundled toolset extensions through the pi loader", async () => {
    const profile = resolveExoProfile("practical");
    const context = {
      agentConfig: { enableAgentToolCreation: false },
    } as TurnContext;
    const tools = new HarnessToolRegistry(context);
    process.env.EXO_PI_EXTENSIONS_BUNDLED =
      "web-tools-extension.ts,todo-tools-extension.ts";
    try {
      await profile.registerTools(tools, context);
      expect(tools.get("web_search")?.source).toBe("library");
      expect(tools.get("web_fetch")?.source).toBe("library");
      expect(tools.get("todowrite")?.source).toBe("library");
      // Extension loading must not break the core toolsets.
      expect(tools.get("snapshot_sandbox")?.source).toBe("built_in");
    } finally {
      delete process.env.EXO_PI_EXTENSIONS_BUNDLED;
    }
  });

  it("exposes legacy agent-tool creation only when enabled", () => {
    const profile = resolveExoProfile("practical");
    const context = {
      agentConfig: { enableAgentToolCreation: true },
    } as TurnContext;

    expect(profile.builtInToolNames(context)).toEqual([
      "shell",
      "inspect_tools",
      "manage_tool",
      "install_agent_tool",
      "uninstall_agent_tool",
    ]);
  });

  it("rejects unknown profiles", () => {
    expect(() => resolveExoProfile("unknown")).toThrow(
      "expected bootstrap or practical",
    );
  });
});
