// The pi-core loop with Exo's shell-only tool surface - kept as an explicit,
// separately-addressable variant purely so benchmark comparisons don't lose
// their shell-only data point now that both exo/harness-pi-core.ts's default
// AND registerExoTools itself (via the practical profile - see git history)
// include file tools.
//
// A plain `fileTools: false` flag isn't enough here: once registerExoTools
// started including write/edit/read on its own, a flag that only skipped a
// *second*, separate registerFileTools call became a no-op - pi-core-bench-*
// agents kept silently getting file tools regardless of the flag, the same
// class of bug exo-harness-shell-only.ts was written to prevent for
// exo-bench-*, just not caught here at the same time. Passing a
// registerTools override instead - one that structurally never imports
// tools/file-tools.ts - closes that gap the same way.
//
// Point pi-core-bench-* agents at this module so they keep meaning what
// their name says - shell-only.

import {
  defineHarness,
  registerBuiltInTools,
  registerLibraryToolModulePath,
} from "@exo/harness";
import { registerConfiguredAgentTools } from "@exo/model-runtime/turn-loop";

import { runPiCoreTurn } from "./harness-pi-core";
import { resolveExoProfile } from "./profiles";
import { registerGuardianTools } from "./tools/guardian-tools";
import { registerSandboxTools } from "./tools/sandbox-tools";

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context, {
      registerTools: async (tools, turnContext) => {
        const profile = resolveExoProfile();
        registerBuiltInTools(
          tools,
          turnContext,
          profile.builtInToolNames(turnContext),
        );
        // Deliberately not profile.registerTools(...): that's where
        // registerFileTools now lives for the practical profile. Register
        // the same profile-owned tools this variant still wants to compare
        // against directly instead - mirrors exo/harness-shell-only.ts.
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
