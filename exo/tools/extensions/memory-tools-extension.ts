// Pi-style extension exposing exo's durable agent memory (remember/forget)
// as a runtime-loadable toolset.

import type { PiExtensionApi } from "../pi-compat";
import { registerMemoryTools } from "../memory-tools";
import { toolInstancesFromRegistrar } from "./helpers";

export default function memoryToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerMemoryTools)) {
    pi.registerToolInstance(tool);
  }
}
