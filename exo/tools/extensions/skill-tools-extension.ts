// Pi-style extension exposing exo's skill tools as a runtime-loadable
// toolset.

import { registerSkillTools } from "@exo/harness";
import type { PiExtensionApi } from "../pi-compat";
import { toolInstancesFromRegistrar } from "./helpers";

export default function skillToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerSkillTools)) {
    pi.registerToolInstance(tool);
  }
}
