/**
 * Streaming handler dispatcher
 */

import type { VertexModelConfig, Context, StreamOptions } from "../types.js";
import type { AssistantMessageEventStream } from "@earendil-works/pi-ai";
import {
  getCurrentSystemMessage,
  getCurrentTools,
  getSystemMessageText,
} from "@earendil-works/pi-ai";
import { streamGemini } from "./gemini.js";
import { streamMaaS } from "./maas.js";
import { installSseCommentFilter } from "./sse-comment-filter.js";

/**
 * Pi 0.99 folds the system prompt and tool definitions into system messages
 * (`toolsAdded`/`toolsRemoved` deltas replayed in order); the agent loop no
 * longer populates `context.systemPrompt`/`context.tools`, which are legacy
 * shorthand. The streaming paths read the shorthand fields, so resolve them
 * from the transcript here. Explicit fields still win when a caller sets them.
 */
function resolveContext(context: Context): Context {
  if (context.systemPrompt !== undefined && context.tools !== undefined) return context;
  const systemMessage = getCurrentSystemMessage(context.messages);
  return {
    ...context,
    systemPrompt: context.systemPrompt ?? (systemMessage ? getSystemMessageText(systemMessage) : undefined),
    tools: context.tools ?? getCurrentTools(context.messages),
  };
}

export function streamVertex(
  model: VertexModelConfig,
  context: Context,
  options?: StreamOptions
): AssistantMessageEventStream {
  // Vertex injects keepalive lines (": keepalive" / "data: : keepalive") on slow
  // streams; SDK parsers choke on them. Strip at the fetch layer (installed once).
  installSseCommentFilter();

  context = resolveContext(context);

  switch (model.endpointType) {
    case "gemini":
      return streamGemini(model, context, options);
    case "maas":
      return streamMaaS(model, context, options);
    default:
      throw new Error(`Unknown endpoint type: ${(model as any).endpointType}`);
  }
}

export { streamGemini, streamMaaS };
