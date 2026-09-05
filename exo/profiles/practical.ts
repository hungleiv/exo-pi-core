import { type BuiltInToolName } from "@exo/harness";

import { registerFileTools } from "../tools/file-tools";
import { registerGuardianTools } from "../tools/guardian-tools";
import { loadPiExtension, piExtensionPathsFromEnv } from "../tools/pi-compat";
import { registerSandboxTools } from "../tools/sandbox-tools";
import type { ExoProfile } from "./types";

// Core toolsets stay profile-owned: they implement the evolution loop
// (sandbox snapshot/rewind), the rebuild/restart guardian, and (as of the
// file-tools pilot's results - see exo/harness-pilot-file-tools.ts and its
// git history) file editing. Everything else ships as pi-style extensions
// under exo/tools/extensions/ and loads
// through EXO_PI_EXTENSIONS / EXO_PI_EXTENSIONS_BUNDLED so the registry can
// reload them on the next turn after edits. The bundled prefix is relative
// to the pi-compat module directory (exo/tools/).
const BUNDLED_EXTENSIONS_DIR = "./extensions";

export const practicalProfile: ExoProfile = {
  name: "practical",
  builtInToolNames(context) {
    const names = bootstrapBuiltInToolNames();
    if (context.agentConfig.enableAgentToolCreation) {
      names.push("install_agent_tool", "uninstall_agent_tool");
    }
    return names;
  },
  async registerTools(tools) {
    registerSandboxTools(tools);
    registerFileTools(tools);
    // Pi-style extensions from EXO_PI_EXTENSIONS (comma-separated paths)
    // and EXO_PI_EXTENSIONS_BUNDLED (comma-separated file names inside
    // exo/tools/extensions/). Unset means nothing extra. The registry is
    // rebuilt every tool round-trip, so edited extensions reload on the
    // next turn.
    for (const extensionPath of piExtensionPathsFromEnv(process.env, {
      bundledPrefix: BUNDLED_EXTENSIONS_DIR,
    })) {
      await loadPiExtension(tools, extensionPath, {
        exposeCommands: true,
      });
    }
    registerGuardianTools(tools);
  },
};

function bootstrapBuiltInToolNames(): BuiltInToolName[] {
  return ["shell", "inspect_tools", "manage_tool"];
}
