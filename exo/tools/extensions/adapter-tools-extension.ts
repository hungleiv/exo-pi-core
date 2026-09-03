// Pi-style extension exposing exo's adapter management tools as a
// runtime-loadable toolset.

import { registerAdapterTools } from "@exo/harness";
import type { PiExtensionApi } from "../pi-compat";
import { toolInstancesFromRegistrar } from "./helpers";

export default function adapterToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerAdapterTools)) {
    pi.registerToolInstance(tool);
  }
}
