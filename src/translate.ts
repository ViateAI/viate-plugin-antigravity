import { randomBytes } from "node:crypto";

export const ANTIGRAVITY_DEFAULT_SYSTEM =
  "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.\n" +
  "You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.\n" +
  "**Absolute paths only**\n" +
  "**Proactiveness**";

const SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator";

type Part = Record<string, unknown>;
type Content = { role: "user" | "model"; parts: Part[] };

export type AntigravityEnvelope = {
  project: string;
  requestId: string;
  request: {
    sessionId: string;
    contents: Content[];
    systemInstruction: { role: "system"; parts: Part[] };
    generationConfig: Record<string, unknown>;
    tools?: unknown[];
    toolConfig?: unknown;
  };
  model: string;
  userAgent: "antigravity";
  requestType: "agent";
};

export function buildAntigravityEnvelope(
  body: Record<string, unknown>,
  projectId: string,
  sessionKey: string,
): AntigravityEnvelope {
  const reasoning = asRecord(body.reasoning);
  const effort = stringValue(body.reasoning_effort) ?? stringValue(reasoning?.effort);
  const model = resolveAntigravityWireModel(cleanModel(stringValue(body.model) ?? ""), effort);
  if (!model) {
    throw Object.assign(new Error("model is required"), { statusCode: 400 });
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const contents = toContents(messages);
  const tools = toTools(body.tools, model.startsWith("claude-"));
  const requestedMax = numberValue(body.max_tokens) ?? numberValue(body.max_completion_tokens);
  const generationConfig: Record<string, unknown> = {
    topK: 40,
    topP: numberValue(body.top_p) ?? 1,
  };
  const temperature = numberValue(body.temperature);
  if (temperature !== undefined) generationConfig.temperature = temperature;
  if (requestedMax !== undefined) generationConfig.maxOutputTokens = Math.floor(requestedMax);
  return {
    project: projectId,
    requestId: `agent/${Date.now()}/${randomBytes(4).toString("hex")}`,
    request: {
      sessionId: sessionId(sessionKey),
      contents,
      systemInstruction: { role: "system", parts: [{ text: ANTIGRAVITY_DEFAULT_SYSTEM }] },
      generationConfig,
      ...(tools ? { tools, toolConfig: toolConfig(body.tool_choice) } : {}),
    },
    model,
    userAgent: "antigravity",
    requestType: "agent",
  };
}

export function antigravityUrl(base: string, stream: boolean): string {
  const trimmed = base.replace(/\/$/, "");
  return stream
    ? `${trimmed}/v1internal:streamGenerateContent?alt=sse`
    : `${trimmed}/v1internal:generateContent`;
}

export function toOpenAIChatCompletion(model: string, payload: unknown): Record<string, unknown> {
  const chunks = asRecords(payload);
  let content = "";
  let reasoning = "";
  const toolCalls: Array<Record<string, unknown>> = [];
  let finish = "stop";
  let usage: Record<string, unknown> | undefined;
  for (const chunk of chunks) {
    const response = unwrap(chunk);
    const candidate = firstCandidate(response);
    finish = finishReason(candidate, finish);
    for (const part of partsOf(candidate)) {
      if (part.thought === true && typeof part.text === "string") {
        reasoning += part.text;
      } else if (typeof part.text === "string") {
        content += part.text;
      } else {
        const image = imageMarkdown(part);
        if (image) content += image;
      }
      const call = functionCall(part);
      if (call) toolCalls.push(openAIToolCall(call, toolCalls.length));
    }
    const nextUsage = openAIUsage(response);
    if (nextUsage) usage = nextUsage;
  }
  if (toolCalls.length > 0 && finish === "stop") finish = "tool_calls";
  const message: Record<string, unknown> = {
    role: "assistant",
    content: content || null,
    ...(reasoning ? { reasoning_content: reasoning } : {}),
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
  return {
    id: `chatcmpl-${randomBytes(8).toString("hex")}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: finish }],
    ...(usage ? { usage } : {}),
  };
}

export type StreamState = {
  id: string;
  created: number;
  nextTool: number;
  finished: boolean;
};

export function createStreamState(): StreamState {
  return {
    id: `chatcmpl-${randomBytes(8).toString("hex")}`,
    created: Math.floor(Date.now() / 1000),
    nextTool: 0,
    finished: false,
  };
}

export function toOpenAIStreamChunks(
  model: string,
  payload: unknown,
  state: StreamState,
): Array<Record<string, unknown>> {
  const chunks: Array<Record<string, unknown>> = [];
  for (const record of asRecords(payload)) {
    const response = unwrap(record);
    const candidate = firstCandidate(response);
    for (const part of partsOf(candidate)) {
      if (part.thought === true && typeof part.text === "string" && part.text) {
        chunks.push(chunk(model, state, { reasoning_content: part.text }, null));
      } else if (typeof part.text === "string" && part.text) {
        chunks.push(chunk(model, state, { content: part.text }, null));
      } else {
        const image = imageMarkdown(part);
        if (image) chunks.push(chunk(model, state, { content: image }, null));
      }
      const call = functionCall(part);
      if (call) {
        const index = state.nextTool;
        state.nextTool += 1;
        chunks.push(chunk(model, state, {
          tool_calls: [{
            index,
            id: call.id,
            type: "function",
            function: { name: call.name, arguments: call.arguments },
          }],
        }, null));
      }
    }
    const reason = candidateFinish(candidate);
    const usage = openAIUsage(response);
    if (reason || usage) {
      state.finished = true;
      const mapped = reason ? mapFinish(reason, state.nextTool > 0) : (state.nextTool > 0 ? "tool_calls" : null);
      chunks.push({
        ...chunk(model, state, {}, mapped),
        ...(usage ? { usage } : {}),
      });
    }
  }
  return chunks;
}

export function encodeSse(frames: Array<Record<string, unknown>>, done = false): Uint8Array {
  const lines = frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`);
  if (done) lines.push("data: [DONE]\n\n");
  return new TextEncoder().encode(lines.join(""));
}

export function takeSseEvents(buffer: string): { events: unknown[]; rest: string } {
  const parts = buffer.split(/\r?\n\r?\n/);
  const rest = parts.pop() ?? "";
  const events: unknown[] = [];
  for (const part of parts) {
    const data = part
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data || data === "[DONE]") continue;
    try {
      events.push(JSON.parse(data) as unknown);
    } catch {
      // Ignore non-JSON SSE comments.
    }
  }
  return { events, rest };
}

function toContents(messages: unknown[]): Content[] {
  const toolText = new Map<string, string>();
  const toolName = new Map<string, string>();
  for (const message of messages) {
    const record = asRecord(message);
    if (!record || record.role !== "tool") continue;
    const id = stringValue(record.tool_call_id);
    if (!id) continue;
    toolText.set(id, contentText(record.content));
    const name = stringValue(record.name);
    if (name) toolName.set(id, name);
  }
  const systemParts: Part[] = [];
  const contents: Content[] = [];
  const consumedTools = new Set<string>();
  for (const message of messages) {
    const record = asRecord(message);
    if (!record || record.role === "tool") continue;
    const role = stringValue(record.role) ?? "";
    if (role === "system" || role === "developer") {
      const text = contentText(record.content);
      if (text) systemParts.push({ text });
      continue;
    }
    if (role === "user") {
      const parts = contentParts(record.content);
      if (parts.length > 0) contents.push({ role: "user", parts });
      continue;
    }
    if (role !== "assistant") continue;
    const parts: Part[] = [];
    const reasoning = stringValue(record.reasoning_content);
    if (reasoning) parts.push({ thought: true, text: reasoning });
    const text = contentText(record.content);
    if (text) parts.push({ text });
    const calls = Array.isArray(record.tool_calls) ? record.tool_calls : [];
    const responseParts: Part[] = [];
    for (const call of calls) {
      const tool = asRecord(call);
      const fn = asRecord(tool?.function);
      if (!tool || tool.type !== "function" || !fn) continue;
      const id = stringValue(tool.id) ?? `call_${parts.length}`;
      const name = sanitizeToolName(stringValue(fn.name) ?? "tool");
      parts.push({
        thoughtSignature: SKIP_THOUGHT_SIGNATURE,
        functionCall: {
          id,
          name,
          args: parseArgs(fn.arguments),
        },
      });
      if (!toolText.has(id)) continue;
      consumedTools.add(id);
      responseParts.push({
        functionResponse: {
          id,
          name: sanitizeToolName(toolName.get(id) ?? name),
          response: { result: parseToolResult(toolText.get(id) ?? "") },
        },
      });
    }
    if (parts.length > 0) contents.push({ role: "model", parts });
    if (responseParts.length > 0) contents.push({ role: "user", parts: responseParts });
  }
  for (const [id, text] of toolText) {
    if (consumedTools.has(id)) continue;
    contents.push({
      role: "user",
      parts: [{ text: `[tool_history_result: ${sanitizeToolName(toolName.get(id) ?? "tool")}] ${text}` }],
    });
  }
  if (systemParts.length > 0) {
    if (contents[0]?.role === "user") contents[0].parts.unshift(...systemParts);
    else contents.unshift({ role: "user", parts: systemParts });
  }
  return mergeSameRole(contents.length > 0 ? contents : [{ role: "user", parts: [{ text: "" }] }]);
}

function toTools(value: unknown, claude = false): unknown[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const declarations: Array<Record<string, unknown>> = [];
  for (const entry of value) {
    const tool = asRecord(entry);
    const fn = asRecord(tool?.function);
    if (!tool || tool.type !== "function" || !fn) continue;
    const name = sanitizeToolName(stringValue(fn.name) ?? "");
    if (!name) continue;
    const parameters = asRecord(fn.parameters);
    declarations.push({
      name,
      ...(stringValue(fn.description) ? { description: stringValue(fn.description) } : {}),
      parameters: parameters ? toolSchema(parameters, claude) : { type: "object", properties: {} },
    });
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
}

function toolSchema(source: Record<string, unknown>, claude: boolean): Record<string, unknown> {
  const schema = geminiSchema(source);
  return claude ? claudeToolSchema(schema) : schema;
}

function geminiSchema(source: Record<string, unknown>): Record<string, unknown> {
  const schema: Record<string, unknown> = {};
  for (const key of ["format", "title", "description", "pattern"] as const) {
    if (typeof source[key] === "string") schema[key] = source[key];
  }
  for (const key of ["default", "example"] as const) {
    if (source[key] !== undefined) schema[key] = source[key];
  }
  for (const key of ["minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"] as const) {
    const numeric = numberValue(source[key]);
    if (numeric !== undefined) schema[key] = numeric;
  }
  const minimum = mergedBound(source.minimum, source.exclusiveMinimum, Math.max);
  const maximum = mergedBound(source.maximum, source.exclusiveMaximum, Math.min);
  if (minimum !== undefined) schema.minimum = minimum;
  if (maximum !== undefined) schema.maximum = maximum;

  const types = schemaTypes(source.type);
  const concrete = types.filter((type) => type !== "null");
  if (source.nullable === true || types.includes("null")) schema.nullable = true;
  if (concrete.length === 1) schema.type = concrete[0];

  if ("const" in source) {
    if (source.const === null) schema.nullable = true;
    else {
      const encoded = enumValue(source.const);
      if (encoded !== undefined) schema.enum = [encoded];
    }
  } else if (Array.isArray(source.enum)) {
    const values = source.enum.map(enumValue).filter((value): value is string => value !== undefined);
    if (values.length > 0) schema.enum = values;
  }

  const properties = asRecord(source.properties);
  if (properties) {
    schema.properties = Object.fromEntries(Object.entries(properties).map(([key, value]) => [key, geminiSchema(asRecord(value) ?? {})]));
  }
  if (Array.isArray(source.required)) {
    const required = source.required.filter((item): item is string => typeof item === "string");
    if (required.length > 0) schema.required = required;
  }
  if (Array.isArray(source.propertyOrdering)) {
    const ordering = source.propertyOrdering.filter((item): item is string => typeof item === "string");
    if (ordering.length > 0) schema.propertyOrdering = ordering;
  }
  if (typeof source.additionalProperties === "boolean") schema.additionalProperties = source.additionalProperties;
  else if (asRecord(source.additionalProperties)) schema.additionalProperties = geminiSchema(asRecord(source.additionalProperties) ?? {});
  if (Array.isArray(source.items)) {
    const items = source.items.map((item) => geminiSchema(asRecord(item) ?? {}));
    if (items.length === 1) schema.items = items[0];
    else if (items.length > 1) schema.items = { anyOf: items };
  } else if (asRecord(source.items)) {
    schema.items = geminiSchema(asRecord(source.items) ?? {});
  }

  const branches: unknown[] = [];
  if (Array.isArray(source.anyOf)) branches.push(...source.anyOf);
  if (concrete.length > 1) branches.push(...concrete.map((type) => ({ type })));
  if (branches.length > 0) schema.anyOf = branches.map((branch) => geminiSchema(asRecord(branch) ?? {}));
  if (typeof source.ref === "string") schema.ref = source.ref;
  const defs = asRecord(source.defs);
  if (defs) schema.defs = Object.fromEntries(Object.entries(defs).map(([key, value]) => [key, geminiSchema(asRecord(value) ?? {})]));
  return schema;
}

// Vertex Claude checks function parameters against a JSON Schema draft 2020-12
// subset and rejects combinators, length bounds, and defaults that Gemini accepts.
// Flatten unions and drop those keywords only on Claude models.
const CLAUDE_SCHEMA_DROP = new Set([
  "minItems",
  "maxItems",
  "minLength",
  "maxLength",
  "minProperties",
  "maxProperties",
  "format",
  "title",
  "default",
  "example",
  "examples",
  "propertyOrdering",
  "ref",
  "defs",
]);

function claudeToolSchema(source: Record<string, unknown>): Record<string, unknown> {
  const flattened = flattenClaudeCombinators(source);
  const stripped = stripClaudeKeywords(flattened);
  return asRecord(stripped) ?? { type: "object", properties: {} };
}

function flattenClaudeCombinators(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(flattenClaudeCombinators);
  const schema = asRecord(node);
  if (!schema) return node;
  const copy: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === "anyOf" || key === "oneOf" || key === "allOf") continue;
    copy[key] = flattenClaudeCombinators(value);
  }
  const combinator = (["anyOf", "oneOf", "allOf"] as const).find((key) => Array.isArray(schema[key]));
  if (!combinator) return copy;
  const branches = (schema[combinator] as unknown[])
    .map(flattenClaudeCombinators)
    .flatMap((branch) => {
      const record = asRecord(branch);
      return record ? [record] : [];
    });
  const nonNull = branches.filter((branch) => branch.type !== "null");
  const merged = mergeClaudeBranches(nonNull.length > 0 ? nonNull : branches);
  const outerDescription = copy.description;
  Object.assign(copy, merged);
  if (typeof outerDescription === "string") copy.description = outerDescription;
  if (nonNull.length !== branches.length) copy.nullable = true;
  return copy;
}

function mergeClaudeBranches(branches: Array<Record<string, unknown>>): Record<string, unknown> {
  const first = branches[0];
  if (!first) return {};
  if (branches.length === 1) return { ...first };
  // `string | string[]` is how clients express "one or many". Keep the string:
  // a single task is the required case (`start` rejects several), and repeated
  // calls still cover the array case. An array-only schema would force `start`
  // to send a list.
  const scalarList = stringOrStringList(branches);
  if (scalarList) return scalarList;
  if (branches.every((branch) => branch.type === "object" || asRecord(branch.properties))) {
    const properties: Record<string, unknown> = {};
    for (const branch of branches) {
      const props = asRecord(branch.properties) ?? {};
      for (const [name, value] of Object.entries(props)) {
        properties[name] = properties[name] === undefined ? value : mergeClaudeProperty(properties[name], value);
      }
    }
    const requiredSets = branches.map((branch) => new Set(
      Array.isArray(branch.required) ? branch.required.filter((item): item is string => typeof item === "string") : [],
    ));
    const required = [...requiredSets[0] ?? []].filter((name) => requiredSets.every((set) => set.has(name)));
    const merged: Record<string, unknown> = { type: "object", properties };
    if (required.length > 0) merged.required = required;
    if (branches.every((branch) => branch.additionalProperties === false)) merged.additionalProperties = false;
    else if (branches.some((branch) => branch.additionalProperties === true)) merged.additionalProperties = true;
    return merged;
  }
  return { ...branches.slice().sort((left, right) => claudeBranchScore(right) - claudeBranchScore(left))[0] ?? {} };
}

function mergeClaudeProperty(left: unknown, right: unknown): unknown {
  const leftRec = asRecord(left);
  const rightRec = asRecord(right);
  if (!leftRec || !rightRec || !leftRec.type || leftRec.type !== rightRec.type) return left;
  const enums = [...new Set([
    ...(Array.isArray(leftRec.enum) ? leftRec.enum : []),
    ...(Array.isArray(rightRec.enum) ? rightRec.enum : []),
  ])];
  return {
    ...leftRec,
    ...rightRec,
    type: leftRec.type,
    ...(enums.length > 0 ? { enum: enums } : {}),
  };
}

function stringOrStringList(branches: Array<Record<string, unknown>>): Record<string, unknown> | undefined {
  if (branches.length !== 2) return undefined;
  const stringBranch = branches.find((branch) => branch.type === "string" && !branch.enum && !branch.const);
  const listBranch = branches.find((branch) => {
    const items = asRecord(branch.items);
    return branch.type === "array" && items?.type === "string";
  });
  return stringBranch && listBranch ? { ...stringBranch } : undefined;
}

function claudeBranchScore(branch: Record<string, unknown>): number {
  if (branch.type === "object" || branch.properties) return 3;
  if (branch.type === "array" || branch.items) return 2;
  if (typeof branch.type === "string" && branch.type !== "null") return 1;
  return 0;
}

function stripClaudeKeywords(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(stripClaudeKeywords);
  const schema = asRecord(node);
  if (!schema) return node;
  const copy: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (CLAUDE_SCHEMA_DROP.has(key)) continue;
    if (key === "properties") {
      const props = asRecord(value);
      copy.properties = props
        ? Object.fromEntries(Object.entries(props).map(([name, child]) => [name, stripClaudeKeywords(child)]))
        : {};
      continue;
    }
    copy[key] = stripClaudeKeywords(value);
  }
  return copy;
}

function schemaTypes(value: unknown): string[] {
  if (typeof value === "string") return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function mergedBound(
  inclusive: unknown,
  exclusive: unknown,
  pick: (left: number, right: number) => number,
): number | undefined {
  const left = numberValue(inclusive);
  const right = numberValue(exclusive);
  if (left === undefined) return right;
  if (right === undefined) return left;
  return pick(left, right);
}

function enumValue(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return undefined;
}

function toolConfig(choice: unknown): Record<string, unknown> {
  if (choice === "none") return { functionCallingConfig: { mode: "NONE" } };
  if (choice === "required") return { functionCallingConfig: { mode: "ANY" } };
  const named = asRecord(choice);
  const fn = asRecord(named?.function);
  const name = stringValue(fn?.name);
  if (named?.type === "function" && name) {
    return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [sanitizeToolName(name)] } };
  }
  return { functionCallingConfig: { mode: "VALIDATED" } };
}

function contentParts(content: unknown): Part[] {
  if (typeof content === "string") return content ? [{ text: content }] : [];
  if (!Array.isArray(content)) return [];
  const parts: Part[] = [];
  for (const entry of content) {
    const part = asRecord(entry);
    if (!part) continue;
    if (part.type === "text" && typeof part.text === "string" && part.text) {
      parts.push({ text: part.text });
    } else if (part.type === "image_url") {
      const image = imagePart(part.image_url);
      if (image) parts.push(image);
    }
  }
  return parts;
}

function imagePart(value: unknown): Part | undefined {
  const record = asRecord(value);
  const url = stringValue(record?.url) ?? stringValue(value);
  if (!url) return undefined;
  const dataUrl = /^data:([^;,]+);base64,([A-Za-z0-9+/=\s]+)$/i.exec(url);
  if (dataUrl?.[1] && dataUrl[2]) {
    return { inlineData: { mimeType: dataUrl[1], data: dataUrl[2].replace(/\s/g, "") } };
  }
  return { fileData: { mimeType: "image/png", fileUri: url } };
}

function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((entry) => {
      const part = asRecord(entry);
      return part?.type === "text" && typeof part.text === "string" ? part.text : "";
    })
    .filter(Boolean)
    .join("");
}

function mergeSameRole(contents: Content[]): Content[] {
  const merged: Content[] = [];
  for (const entry of contents) {
    const last = merged[merged.length - 1];
    if (last && last.role === entry.role) last.parts.push(...entry.parts);
    else merged.push({ role: entry.role, parts: [...entry.parts] });
  }
  return merged.filter((entry) => entry.parts.length > 0);
}

function unwrap(payload: Record<string, unknown>): Record<string, unknown> {
  return asRecord(payload.response) ?? payload;
}

function firstCandidate(response: Record<string, unknown>): Record<string, unknown> {
  const candidates = Array.isArray(response.candidates) ? response.candidates : [];
  return asRecord(candidates[0]) ?? {};
}

function partsOf(candidate: Record<string, unknown>): Part[] {
  const content = asRecord(candidate.content);
  return Array.isArray(content?.parts) ? content.parts.flatMap((part) => asRecord(part) ? [part] : []) : [];
}

function functionCall(part: Part): { id: string; name: string; arguments: string } | undefined {
  const call = asRecord(part.functionCall);
  const name = stringValue(call?.name);
  if (!call || !name) return undefined;
  return {
    id: stringValue(call.id) ?? `call_${sanitizeToolName(name)}`,
    name,
    arguments: JSON.stringify(call.args ?? {}),
  };
}

function imageMarkdown(part: Part): string {
  const inline = asRecord(part.inlineData);
  const data = typeof inline?.data === "string" ? inline.data : "";
  if (data) {
    const mime = typeof inline?.mimeType === "string" ? inline.mimeType : "image/png";
    return `\n![image](data:${mime};base64,${data})`;
  }
  const file = asRecord(part.fileData);
  const uri = typeof file?.fileUri === "string" ? file.fileUri : "";
  return uri ? `\n![image](${uri})` : "";
}

function openAIToolCall(
  call: { id: string; name: string; arguments: string },
  index: number,
): Record<string, unknown> {
  return {
    id: call.id,
    index,
    type: "function",
    function: { name: call.name, arguments: call.arguments },
  };
}

function candidateFinish(candidate: Record<string, unknown>): string | undefined {
  return stringValue(candidate.finishReason);
}

function finishReason(candidate: Record<string, unknown>, current: string): string {
  const reason = candidateFinish(candidate);
  return reason ? mapFinish(reason, false) : current;
}

function mapFinish(reason: string, hasTools: boolean): string {
  const upper = reason.toUpperCase();
  if (upper === "MAX_TOKENS") return "length";
  if (upper === "SAFETY" || upper === "RECITATION") return "content_filter";
  if (hasTools) return "tool_calls";
  return "stop";
}

function openAIUsage(response: Record<string, unknown>): Record<string, unknown> | undefined {
  const usage = asRecord(response.usageMetadata);
  if (!usage) return undefined;
  const prompt = numberValue(usage.promptTokenCount) ?? 0;
  const thoughts = numberValue(usage.thoughtsTokenCount) ?? 0;
  let candidates = numberValue(usage.candidatesTokenCount) ?? 0;
  const total = numberValue(usage.totalTokenCount);
  if (candidates === 0 && total !== undefined) {
    candidates = Math.max(0, total - prompt - thoughts);
  }
  const cached = numberValue(usage.cachedContentTokenCount) ?? 0;
  return {
    prompt_tokens: prompt,
    completion_tokens: candidates + thoughts,
    total_tokens: total ?? prompt + candidates + thoughts,
    ...(cached > 0 ? { prompt_tokens_details: { cached_tokens: cached } } : {}),
    ...(thoughts > 0 ? { completion_tokens_details: { reasoning_tokens: thoughts } } : {}),
  };
}

function chunk(
  model: string,
  state: StreamState,
  delta: Record<string, unknown>,
  finish: string | null,
): Record<string, unknown> {
  return {
    id: state.id,
    object: "chat.completion.chunk",
    created: state.created,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

function sessionId(key: string): string {
  const FNV_OFFSET = -3750763034362895579n;
  const FNV_PRIME = 1099511628211n;
  let hash = FNV_OFFSET;
  for (const byte of Buffer.from(key || "antigravity", "utf8")) {
    hash = BigInt.asIntN(64, hash ^ BigInt(byte));
    hash = BigInt.asIntN(64, hash * FNV_PRIME);
  }
  return hash.toString();
}

function cleanModel(model: string): string {
  const slash = model.lastIndexOf("/");
  return (slash >= 0 ? model.slice(slash + 1) : model).trim();
}

const HYPHEN_EFFORTS = ["-extra-high", "-xhigh", "-minimal", "-medium", "-none", "-high", "-low", "-max"] as const;
const TIER_PREFERENCE = ["low", "medium", "high", "minimal", "xhigh", "max", "none"];
const tieredEfforts = new Map<string, string[]>();

type AntigravityFamily = { stem: string; effort?: string; fast: boolean };

function parseAntigravityFamily(modelId: string): AntigravityFamily | null {
  const id = modelId.trim().toLowerCase();
  const fast = id.endsWith("-fast");
  const rest = fast ? id.slice(0, -"-fast".length) : id;
  if (!rest) return null;
  for (const tail of HYPHEN_EFFORTS) {
    if (rest.endsWith(tail) && rest.length > tail.length) {
      const effort = tail.slice(1) === "extra-high" ? "xhigh" : tail.slice(1);
      return { stem: rest.slice(0, -tail.length), effort, fast };
    }
  }
  if (!fast) return null;
  return { stem: rest, fast };
}

function listedEffort(effort: string | undefined): string | undefined {
  if (!effort) return undefined;
  const level = effort.trim().toLowerCase();
  if (!level || level === "off") return undefined;
  if (level === "extra-high") return "xhigh";
  return (TIER_PREFERENCE as readonly string[]).includes(level) ? level : undefined;
}

function defaultListedTier(seen: readonly string[]): string | undefined {
  for (const tier of TIER_PREFERENCE) {
    if (seen.includes(tier)) return tier;
  }
  return seen[0];
}

function cleanEffortLabel(label: string | undefined): string | undefined {
  if (!label) return undefined;
  const efforts = "extra-high|extra high|xhigh|minimal|medium|none|high|low|max";
  const text = label
    .trim()
    .replace(new RegExp(`(?:[\\s\\-_(]+(?:${efforts})\\)?)+$`, "i"), "")
    .replace(/[\s\-_]+$/g, "")
    .trim();
  return text || undefined;
}

function thinkingSuffixStem(stem: string): boolean {
  return stem.startsWith("gemini-") || stem.startsWith("claude-");
}

export function collapseAntigravityModels<T extends { id: string; name?: string }>(models: T[]): T[] {
  const efforts = new Map<string, string[]>();
  const families = models.map((model) => ({ model, family: parseAntigravityFamily(model.id) }));
  for (const item of families) {
    if (!item.family?.effort) continue;
    const logical = `${item.family.stem}${item.family.fast ? "-fast" : ""}`;
    const tiers = efforts.get(logical) ?? [];
    if (!tiers.includes(item.family.effort)) tiers.push(item.family.effort);
    efforts.set(logical, tiers);
  }
  const collapsing: Record<string, true> = {};
  for (const [logical, tiers] of efforts) {
    const stem = logical.endsWith("-fast") ? logical.slice(0, -"-fast".length) : logical;
    // gpt-oss-120b-medium is a size, not a thinking level. A single non-Gemini/Claude
    // suffix stays as its own id; two or more tiers are a thinking family.
    if (stem.startsWith("gpt-oss")) continue;
    if (thinkingSuffixStem(stem) || tiers.length >= 2) collapsing[logical] = true;
  }
  tieredEfforts.clear();
  for (const logical of Object.keys(collapsing)) {
    const tiers = efforts.get(logical);
    if (tiers) tieredEfforts.set(logical, tiers);
  }
  const collapsed: T[] = [];
  const seen: Record<string, true> = {};
  for (const item of families) {
    const logical = item.family ? `${item.family.stem}${item.family.fast ? "-fast" : ""}` : item.model.id;
    const id = item.family?.effort && collapsing[logical] ? logical : item.model.id;
    if (seen[id]) continue;
    seen[id] = true;
    if (id === item.model.id) {
      collapsed.push(item.model);
      continue;
    }
    const name = cleanEffortLabel(item.model.name);
    collapsed.push({ ...item.model, id, ...(name && name !== id ? { name } : { name: undefined }) });
  }
  return collapsed;
}

export function resolveAntigravityWireModel(modelId: string, effort?: string): string {
  const bare = modelId.trim();
  const key = bare.toLowerCase();
  const family = parseAntigravityFamily(key);
  const logical = family ? `${family.stem}${family.fast ? "-fast" : ""}` : key;
  const seen = tieredEfforts.get(logical);
  const suffixFamily = Boolean(family?.effort && thinkingSuffixStem(family.stem));
  if (!suffixFamily && !seen) return bare;
  const tier = listedEffort(effort) ?? family?.effort ?? defaultListedTier(seen ?? []);
  if (!tier) return bare;
  return `${family?.stem ?? logical}-${tier}${family?.fast ? "-fast" : ""}`;
}

function sanitizeToolName(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/g, "_").replace(/^[^A-Za-z_]+/, "");
  return cleaned || "tool";
}

function parseArgs(value: unknown): unknown {
  if (typeof value !== "string") return value ?? {};
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return { input: value };
  }
}

function parseToolResult(value: string): unknown {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" ? parsed : { result: parsed };
  } catch {
    return { result: value };
  }
}

function asRecords(payload: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(payload)) return payload.flatMap((entry) => asRecord(entry) ? [entry] : []);
  const record = asRecord(payload);
  return record ? [record] : [];
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
