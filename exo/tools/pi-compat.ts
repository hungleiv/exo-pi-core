// Pi-style extension compatibility for the Exo harness.
//
// Pi (pi.dev) keeps its core tiny and grows through TypeScript extensions: a
// default-exported function that receives an API object and calls
// registerTool / registerCommand / on. This module lets Exo consume that same
// shape: load a pi-style extension file, convert each registered tool to an
// Exo ToolInstance, and register the result in a HarnessToolRegistry.
//
// What maps cleanly:
//   pi.registerTool  -> HarnessToolRegistry.register (source "library")
//   pi tool parameters (JSON Schema) -> strict-mode JSON Schema
//     (additionalProperties false, every property key in required,
//     optional properties expressed as nullable)
//   pi.registerCommand -> one dispatcher tool (opt-in via exposeCommands)
//   pi.get/setActiveTools -> tracked on the loaded extension record
//   pi.on("tool_call", ...) -> wrapped around every converted handler
//
// Other pi.on(...) subscriptions (session_start/shutdown, ...) have no
// registry equivalent and are reported as unsupported so callers can port
// them by hand.
//
// Execution model: converted tools run as library tools in the harness
// runner process (same as web-tools/memory-tools). The `route` option only
// declares which filesystem scope a tool operates on ("host" workspace vs
// the agent "sandbox" mount) and selects the default cwd handed to the pi
// execute function. It does not sandbox the JS itself; JS sandboxing in
// Exo goes through the shell/sandbox tool boundary.

import { statSync } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import type {
  HarnessToolRegistry,
  JsonObject,
  JsonValue,
  ToolInstance,
} from "@exo/harness";

export type PiToolRoute = "host" | "sandbox";

export interface PiToolContext {
  route: PiToolRoute;
  cwd: string;
  extensionName: string;
}

export interface PiToolDefinition {
  name: string;
  description: string;
  parameters?: JsonValue;
  execute: (
    args: JsonObject,
    ctx: PiToolContext,
  ) => Promise<JsonValue> | JsonValue;
}

export interface PiCommandDefinition {
  name: string;
  description: string;
  execute: (args: string, ctx: PiToolContext) => Promise<JsonValue> | JsonValue;
}

export type PiToolCallListener = (event: {
  toolName: string;
  args: JsonObject;
}) => void;

export interface PiExtensionApi {
  registerTool: (tool: PiToolDefinition) => void;
  registerCommand: (command: PiCommandDefinition) => void;
  // Register an already-complete Exo ToolInstance (definition + handler with
  // the native `execute(args, execution)` signature). Lets an extension wrap
  // existing exo toolsets — execution flows through unchanged, so handlers
  // keep their access to the TurnContext.
  registerToolInstance: (tool: ToolInstance) => void;
  getActiveTools: () => string[];
  setActiveTools: (names: string[]) => void;
  on: (event: string, listener: PiToolCallListener) => void;
}

export type PiExtensionModule = (api: PiExtensionApi) => void | Promise<void>;

export interface LoadPiExtensionOptions {
  route?: PiToolRoute;
  cwd?: string;
  exposeCommands?: boolean;
  // Append ?v=<mtime> so edited files reload on the next turn. Disable in
  // test runners whose transform pipeline cannot handle query imports.
  cacheBust?: boolean;
}

export interface LoadedPiExtension {
  name: string;
  path: string;
  tools: ToolInstance[];
  unsupportedEvents: string[];
}

// Read EXO_PI_EXTENSIONS (comma-separated extension file paths) plus
// EXO_PI_EXTENSIONS_BUNDLED (comma-separated file names resolved against
// this module's ../extensions directory). Empty or unset means no pi
// extensions; the profile registers nothing extra.
export interface PiExtensionPathsOptions {
  // Resolve EXO_PI_EXTENSIONS_BUNDLED entries against this directory,
  // interpreted relative to this module's own directory. Entries starting
  // with "/" are treated as absolute. Use EXO_PI_EXTENSIONS for arbitrary
  // paths.
  bundledPrefix?: string;
}

export function piExtensionPathsFromEnv(
  env: NodeJS.ProcessEnv = process.env,
  options: PiExtensionPathsOptions = {},
): string[] {
  const raw = env.EXO_PI_EXTENSIONS ?? "";
  const bundledRaw = env.EXO_PI_EXTENSIONS_BUNDLED ?? "";
  const prefix = options.bundledPrefix;
  return [
    ...splitCommaList(raw),
    ...splitCommaList(bundledRaw).map((entry) =>
      prefix && !path.isAbsolute(entry)
        ? joinExtensionPath(prefix, entry)
        : entry,
    ),
  ];
}

function splitCommaList(raw: string): string[] {
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

// Resolve `prefix/entry` against the directory of this module, so a caller
// passing bundledPrefix "../tools/extensions" + entry "web-tools-extension.ts"
// gets <this module dir>/../tools/extensions/web-tools-extension.ts.
function joinExtensionPath(prefix: string, entry: string): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, prefix, entry);
}

const SUPPORTED_EVENTS = new Set(["tool_call"]);

// Build the module URL for an extension file. With cache busting, the file
// mtime rides in a query string so the ESM loader treats each saved version
// as a new module (verified under tsx; vitest's transform pipeline cannot
// handle query imports, so tests disable it).
export function piExtensionModuleUrl(
  extensionPath: string,
  cacheBust = true,
): string {
  const base = pathToFileURL(extensionPath).href;
  if (!cacheBust) {
    return base;
  }
  const mtimeMs =
    statSync(extensionPath, { throwIfNoEntry: false })?.mtimeMs ?? 0;
  return `${base}?v=${mtimeMs}`;
}

function extensionNameFromPath(extensionPath: string): string {
  const base = path.basename(extensionPath).replace(/\.[^.]+$/, "");
  return (
    base.replace(/[^A-Za-z0-9_.-]+/g, "_") || `ext_${randomUUID().slice(0, 8)}`
  );
}

function isRecord(value: JsonValue | unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Normalize a pi-style JSON Schema object into the strict-mode shape Exo
// tool definitions use: no additional properties, every declared property
// key listed in required, optionals expressed as nullable.
export function toStrictParameters(
  parameters: JsonValue | undefined,
): JsonObject {
  if (!isRecord(parameters)) {
    return {
      type: "object",
      additionalProperties: false,
      properties: {},
      required: [],
    };
  }
  const properties: JsonObject = isRecord(parameters.properties)
    ? { ...parameters.properties }
    : {};
  const declaredRequired: string[] = Array.isArray(parameters.required)
    ? parameters.required.filter(
        (entry): entry is string => typeof entry === "string",
      )
    : [];
  const required = Object.keys(properties);
  for (const [key, schema] of Object.entries(properties)) {
    if (declaredRequired.includes(key) || !isRecord(schema)) {
      continue;
    }
    const type = schema.type;
    if (typeof type === "string") {
      properties[key] = { ...schema, type: [type, "null"] };
    }
  }
  return {
    type: "object",
    additionalProperties: false,
    properties,
    required,
  };
}

export async function loadPiExtension(
  registry: HarnessToolRegistry,
  extensionPath: string,
  options: LoadPiExtensionOptions = {},
): Promise<LoadedPiExtension> {
  const route = options.route ?? "host";
  const cwd = options.cwd ?? process.cwd();
  const name = extensionNameFromPath(extensionPath);

  const tools: PiToolDefinition[] = [];
  const commands: PiCommandDefinition[] = [];
  const toolInstances: ToolInstance[] = [];
  const toolCallListeners: PiToolCallListener[] = [];
  const unsupportedEvents: string[] = [];
  let activeTools: string[] | null = null;

  const api: PiExtensionApi = {
    registerTool: (tool) => {
      tools.push(tool);
    },
    registerCommand: (command) => {
      commands.push(command);
    },
    registerToolInstance: (tool) => {
      toolInstances.push(tool);
    },
    getActiveTools: () =>
      activeTools ?? [
        ...tools.map((tool) => tool.name),
        ...toolInstances.map((tool) => tool.definition.name),
      ],
    setActiveTools: (names) => {
      activeTools = [...names];
    },
    on: (event, listener) => {
      if (SUPPORTED_EVENTS.has(event)) {
        toolCallListeners.push(listener);
        return;
      }
      unsupportedEvents.push(event);
    },
  };

  // Cache-bust by mtime so editing an extension file takes effect on the
  // next turn without restarting the harness (pi's /reload semantics; the
  // registry is rebuilt every tool round-trip). Auto-disabled under vitest,
  // whose transform pipeline cannot handle query imports.
  const moduleUrl = piExtensionModuleUrl(
    extensionPath,
    options.cacheBust ?? process.env.VITEST === undefined,
  );
  const module = (await import(moduleUrl)) as unknown as { default?: unknown };
  if (typeof module.default !== "function") {
    throw new Error(`pi extension has no default export: ${extensionPath}`);
  }
  await (module.default as PiExtensionModule)(api);

  const ctx: PiToolContext = { route, cwd, extensionName: name };
  const instances: ToolInstance[] = tools.map((tool) =>
    toToolInstance(tool, ctx, toolCallListeners),
  );
  // Pre-built instances keep their native handler; tool_call listeners wrap
  // them the same way so extensions can observe every converted tool.
  instances.push(
    ...toolInstances.map((tool) =>
      wrapToolInstanceListeners(tool, toolCallListeners),
    ),
  );

  if (options.exposeCommands && commands.length > 0) {
    instances.push(toCommandDispatcher(name, commands, ctx));
  }

  for (const instance of instances) {
    registry.register(instance);
  }

  return { name, path: extensionPath, tools: instances, unsupportedEvents };
}

function toToolInstance(
  tool: PiToolDefinition,
  ctx: PiToolContext,
  listeners: PiToolCallListener[],
): ToolInstance {
  return {
    source: "library",
    definition: {
      name: tool.name,
      description: tool.description,
      parameters: toStrictParameters(tool.parameters),
    },
    handler: {
      async execute(args) {
        for (const listener of listeners) {
          listener({ toolName: tool.name, args });
        }
        const result = await tool.execute(args, ctx);
        return result as JsonValue;
      },
    },
  };
}

// Wrap a native ToolInstance's handler so tool_call listeners fire for it
// too, while passing `execution` (and its TurnContext) through untouched.
// Instances arriving through an extension are classified as library tools
// regardless of the source the original toolset declared.
function wrapToolInstanceListeners(
  tool: ToolInstance,
  listeners: PiToolCallListener[],
): ToolInstance {
  const inner = tool.handler;
  if (listeners.length === 0) {
    return { ...tool, source: "library" };
  }
  return {
    ...tool,
    source: "library",
    handler: {
      async execute(args, execution) {
        for (const listener of listeners) {
          listener({ toolName: tool.definition.name, args });
        }
        return inner.execute(args, execution);
      },
    },
  };
}

function toCommandDispatcher(
  extensionName: string,
  commands: PiCommandDefinition[],
  ctx: PiToolContext,
): ToolInstance {
  const byName = new Map(commands.map((command) => [command.name, command]));
  return {
    source: "library",
    definition: {
      name: `${extensionName}_command`,
      description: `Dispatch a /command from the ${extensionName} pi extension.`,
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          command: {
            type: "string",
            description: "Command name to run.",
            enum: commands.map((command) => command.name),
          },
          args: {
            type: ["string", "null"],
            description: "Raw argument text for the command.",
          },
        },
        required: ["command", "args"],
      },
    },
    handler: {
      async execute(args) {
        const commandName = args.command;
        if (typeof commandName !== "string" || !byName.has(commandName)) {
          return {
            ok: false,
            error: `unknown command: ${String(commandName)}`,
          };
        }
        const rawArgs = typeof args.args === "string" ? args.args : "";
        const result = await byName.get(commandName)?.execute(rawArgs, ctx);
        return result as JsonValue;
      },
    },
  };
}
