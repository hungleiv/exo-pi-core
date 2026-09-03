import {
  HarnessToolRegistry,
  registerAdapterTools,
  registerSkillTools,
  type BuiltInToolName,
} from "@exo/harness";

import { registerGuardianTools } from "../tools/guardian-tools";
import { registerIntrospectionTools } from "../tools/introspection-tools";
import { registerMemoryTools } from "../tools/memory-tools";
import { loadPiExtension, piExtensionPathsFromEnv } from "../tools/pi-compat";
import { registerSandboxTools } from "../tools/sandbox-tools";
import { registerSchedulerTools } from "../tools/scheduler-tools";
import { registerTodoTools } from "../tools/todo-tools";
import { registerWebTools } from "../tools/web-tools";
import type { ExoProfile } from "./types";

export const practicalProfile: ExoProfile = {
  name: "practical",
  builtInToolNames(context) {
    const names = bootstrapBuiltInToolNames();
    if (context.agentConfig.enableAgentToolCreation) {
      names.push("install_agent_tool", "uninstall_agent_tool");
    }
    return names;
  },
  async registerTools(tools, context) {
    const libraryTools = new HarnessToolRegistry(context);
    registerSchedulerTools(libraryTools);
    registerAdapterTools(libraryTools);
    registerIntrospectionTools(libraryTools);
    registerSandboxTools(libraryTools);
    registerMemoryTools(libraryTools);
    registerTodoTools(libraryTools);
    registerSkillTools(libraryTools);
    registerWebTools(libraryTools);
    // Pi-style extensions from EXO_PI_EXTENSIONS (comma-separated paths).
    // Unset means nothing extra. The registry is rebuilt every tool
    // round-trip, so edited extensions reload on the next turn.
    for (const extensionPath of piExtensionPathsFromEnv()) {
      await loadPiExtension(libraryTools, extensionPath, {
        exposeCommands: true,
      });
    }
    for (const tool of libraryTools.instances()) {
      tools.register({ ...tool, source: "library" });
    }
    registerGuardianTools(tools);
  },
};

function bootstrapBuiltInToolNames(): BuiltInToolName[] {
  return ["shell", "inspect_tools", "manage_tool"];
}
