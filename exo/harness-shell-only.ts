// The real "exo" harness tool composition, minus file tools - kept as an
// explicit, separately-addressable comparison point now that
// registerFileTools became unconditional in the practical profile's
// registerTools (see git history for that commit and the pilot that led to
// it). Before that change, exo/harness.ts itself WAS the shell-only
// baseline every "exo-bench-*" benchmark agent measured; after it, every
// agent pointed at exo/harness.ts silently gained write/edit/read, which
// would have made future exo-bench-* runs measure a different harness than
// their name and this repo's benchmark history say they do.
//
// Point exo-bench-* agents here instead of at exo/harness.ts to keep
// getting that comparison point. Mirrors registerExoTools's composition
// exactly (registerBuiltInTools, profile.registerTools, library tool
// modules, registerConfiguredAgentTools) with registerFileTools's effect
// undone by simply not being in the chain - this module never imports
// tools/file-tools.ts at all, so there is nothing to omit-by-flag.

import {
  defineHarness,
  registerBuiltInTools,
  registerLibraryToolModulePath,
} from "@exo/harness";
import {
  registerConfiguredAgentTools,
  runResponsesHarnessTurn,
} from "@exo/model-runtime/turn-loop";

import { exoInstructions } from "./harness";
import { resolveExoProfile } from "./profiles";
import { registerGuardianTools } from "./tools/guardian-tools";
import { registerSandboxTools } from "./tools/sandbox-tools";

export default defineHarness({
  async runTurn(context) {
    await runResponsesHarnessTurn(context, {
      instructions: exoInstructions,
      registerTools: async (tools, turnContext) => {
        const profile = resolveExoProfile();
        registerBuiltInTools(
          tools,
          turnContext,
          profile.builtInToolNames(turnContext),
        );
        // Deliberately not profile.registerTools(...): that's where
        // registerFileTools now lives for the practical profile. Register
        // the same profile-owned tools this harness still wants to compare
        // against directly instead.
        registerSandboxTools(tools);
        registerGuardianTools(tools);
        for (const modulePath of turnContext.agentConfig.typescript
          ?.toolModulePaths ?? []) {
          await registerLibraryToolModulePath(tools, turnContext, modulePath);
        }
        await registerConfiguredAgentTools(tools, turnContext);
      },
    });
  },
});
