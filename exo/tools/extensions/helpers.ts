// Shared helpers for the bundled exo toolset extensions. An extension that
// wraps an existing exo toolset registrar (register<Name>Tools) uses
// toolInstancesFromRegistrar to materialize its ToolInstances without a
// real TurnContext: registration only builds definitions and handlers, and
// the actual TurnContext arrives per-call through `execution.context` at
// execute time.

import { HarnessToolRegistry, type TurnContext } from "@exo/harness";
import type { ToolInstance } from "@exo/harness";

export type ToolRegistrar = (registry: HarnessToolRegistry) => void;

// Run a register<Name>Tools registrar against a throwaway registry and
// return the registered instances. The registry's context is only used at
// execute time (passed through by the harness), so a stub is fine here.
export function toolInstancesFromRegistrar(
  registrar: ToolRegistrar,
): ToolInstance[] {
  const registry = new HarnessToolRegistry({} as TurnContext);
  registrar(registry);
  return registry.instances();
}
