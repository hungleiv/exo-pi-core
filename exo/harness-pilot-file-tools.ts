// Pilot: the real "exo" harness (exo/harness.ts's exact tool set and
// instructions - registerExoTools/exoInstructions, unchanged) with
// write/edit/read added on top, scoped to exactly one agent via --module
// rather than touching built-in-tools.ts or the practical profile.
//
// This exists to answer the open question from reviewing file-tools.ts
// against built-in-tools.ts's conventions (see that review's commit): the
// code is now hardened and tested, but has never run as part of a real
// agent's actual tool set, only inside the pi-core prototype loop. Before
// registerBuiltInTools gets a fourth entry that every production "exo"
// agent picks up, one real agent should run it for a while under
// exo-monitor's watch.
//
// Not wired into any profile and not the default for anything - an agent
// only gets this by pointing --module here explicitly.

import { defineHarness } from "@exo/harness";
import { runResponsesHarnessTurn } from "@exo/model-runtime/turn-loop";

import { exoInstructions, registerExoTools } from "./harness";
import { FILE_TOOLS_INSTRUCTION, registerFileTools } from "./tools/file-tools";

export default defineHarness({
  async runTurn(context) {
    await runResponsesHarnessTurn(context, {
      instructions: async (turnContext, tools) => {
        const instructions = await exoInstructions(turnContext, tools);
        instructions.push({
          role: "developer",
          content: FILE_TOOLS_INSTRUCTION,
        });
        return instructions;
      },
      registerTools: async (tools, turnContext) => {
        await registerExoTools(tools, turnContext);
        registerFileTools(tools);
      },
    });
  },
});
