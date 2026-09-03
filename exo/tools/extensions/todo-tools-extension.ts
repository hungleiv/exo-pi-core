// Pi-style extension exposing exo's conversation-scoped todo tracking as a
// runtime-loadable toolset.

import type { PiExtensionApi } from "../pi-compat";
import { registerTodoTools } from "../todo-tools";
import { toolInstancesFromRegistrar } from "./helpers";

export default function todoToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerTodoTools)) {
    pi.registerToolInstance(tool);
  }
}
