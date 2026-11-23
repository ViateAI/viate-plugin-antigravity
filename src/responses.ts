import { createStreamState, takeSseEvents, toOpenAIStreamChunks } from "./translate.js";

type Rejected = { ok: false; status: number; message: string };

export type ResponsesChat =
  | { ok: true; chat: Record<string, unknown> }
  | Rejected;

type ChatMessage = Record<string, unknown>;
type Piece =
  | { kind: "reasoning"; id: string; text: string; done: boolean }
  | { kind: "message"; id: string; text: string; done: boolean }
  | { kind: "tool"; id: string; callId: string; name: string; arguments: string; done: boolean };

export type ResponsesStreamState = {
  id: string;
  createdAt: number;
  model: string;
  seq: number;
  started: boolean;
  completed: boolean;
  pieces: Piece[];
  toolAt: Map<number, number>;
  usage?: Record<string, unknown>;
  status: "completed" | "incomplete";
  incompleteReason?: string;
};

export function responsesRequestToChat(body: Record<string, unknown>): ResponsesChat {
  if (present(body.previous_response_id)) {
    return reject(400, "previous_response_id is not supported; send the full input");
  }
  if (body.conversation != null) {
    return reject(400, "conversation state is not supported; send the full input");
  }
  if (body.background === true) {
    return reject(400, "background responses are not supported");
  }
  if (body.store === true) {
    return reject(400, "store is not supported; send the full input");
  }
  const model = stringValue(body.model);
  if (!model) return reject(400, "model is required");
  const messages: ChatMessage[] = [];
  const instructions = stringValue(body.instructions);
  if (instructions) messages.push({ role: "system", content: instructions });
  const input = readInput(body.input);
  if (!input.ok) return input;
  messages.push(...input.messages);
  const tools = readTools(body.tools);
  if (!tools.ok) return tools;
  const toolChoice = readToolChoice(body.tool_choice);
  if (!toolChoice.ok) return toolChoice;
  const chat: Record<string, unknown> = { model, messages };
  if (body.stream === true) chat.stream = true;
  const maxOutput = numberValue(body.max_output_tokens);
  if (maxOutput !== undefined) chat.max_tokens = maxOutput;
  const temperature = numberValue(body.temperature);
  if (temperature !== undefined) chat.temperature = temperature;
  const topP = numberValue(body.top_p);
  if (topP !== undefined) chat.top_p = topP;
  if (tools.tools) chat.tools = tools.tools;
  if (toolChoice.choice !== undefined) chat.tool_choice = toolChoice.choice;
  const effort = stringValue(body.reasoning_effort) ?? stringValue(record(body.reasoning)?.effort);
  if (effort) chat.reasoning_effort = effort;
  return { ok: true, chat };
}

export function chatCompletionToResponse(model: string, completion: unknown): Record<string, unknown> {
  const state = createResponsesStream(model);
  const root = record(completion);
  const choice = record(arrayAt(root?.choices, 0));
  const message = record(choice?.message);
  consumeChatChunks([{
    choices: [{
      delta: {
        ...(typeof message?.reasoning_content === "string" ? { reasoning_content: message.reasoning_content } : {}),
        ...(typeof message?.content === "string" ? { content: message.content } : {}),
        ...(Array.isArray(message?.tool_calls) ? { tool_calls: message.tool_calls } : {}),
      },
      finish_reason: typeof choice?.finish_reason === "string" ? choice.finish_reason : "stop",
    }],
    ...(root?.usage ? { usage: root.usage } : {}),
  }], state);
  finishResponsesStream(state);
  return snapshot(state);
}

export function createResponsesStream(model: string): ResponsesStreamState {
  const state: ResponsesStreamState = {
    id: "",
    createdAt: Math.floor(Date.now() / 1000),
    model,
    seq: 0,
    started: false,
    completed: false,
    pieces: [],
    toolAt: new Map(),
    status: "completed",
  };
  state.id = nextId(state, "resp");
  return state;
}

export function consumeChatChunks(chunks: unknown[], state: ResponsesStreamState): Array<Record<string, unknown>> {
  const events: Array<Record<string, unknown>> = [];
  for (const chunk of chunks) {
    const root = record(chunk);
    if (!root) continue;
    events.push(...ensureStarted(state));
    const mapped = responsesUsage(record(root.usage));
    if (mapped) state.usage = mapped;
    const choice = record(arrayAt(root.choices, 0));
    const delta = record(choice?.delta) ?? {};
    if (typeof delta.reasoning_content === "string" && delta.reasoning_content) {
      events.push(...appendReasoning(state, delta.reasoning_content));
    }
    if (typeof delta.content === "string" && delta.content) {
      events.push(...appendText(state, delta.content));
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) events.push(...appendTool(state, call));
    }
    if (typeof choice?.finish_reason === "string") applyFinish(state, choice.finish_reason);
  }
  return events;
}

export function finishResponsesStream(state: ResponsesStreamState): Array<Record<string, unknown>> {
  if (state.completed) return [];
  const events = ensureStarted(state);
  events.push(...closeOpen(state));
  state.completed = true;
  events.push({
    type: state.status === "incomplete" ? "response.incomplete" : "response.completed",
    response: snapshot(state),
  });
  return events;
}

export function encodeResponseEvents(events: Array<Record<string, unknown>>): Uint8Array {
  const lines = events.map((event) => {
    const type = typeof event.type === "string" ? event.type : "response";
    return `event: ${type}\ndata: ${JSON.stringify(event)}\n\n`;
  });
  return new TextEncoder().encode(lines.join(""));
}

export function streamResponses(
  body: ReadableStream<Uint8Array> | null,
  model: string,
): ReadableStream<Uint8Array> | null {
  if (!body) return null;
  const chatState = createStreamState();
  const responseState = createResponsesStream(model);
  let pending = "";
  const decoder = new TextDecoder();
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      pending += decoder.decode(chunk, { stream: true });
      const taken = takeSseEvents(pending);
      pending = taken.rest;
      const frames = taken.events.flatMap((event) => toOpenAIStreamChunks(model, event, chatState));
      const encoded = encodeResponseEvents(consumeChatChunks(frames, responseState));
      if (encoded.byteLength > 0) controller.enqueue(encoded);
    },
    flush(controller) {
      pending += decoder.decode();
      if (pending.trim()) {
        const taken = takeSseEvents(`${pending}\n\n`);
        const frames = taken.events.flatMap((event) => toOpenAIStreamChunks(model, event, chatState));
        const encoded = encodeResponseEvents(consumeChatChunks(frames, responseState));
        if (encoded.byteLength > 0) controller.enqueue(encoded);
      }
      const tail = encodeResponseEvents(finishResponsesStream(responseState));
      if (tail.byteLength > 0) controller.enqueue(tail);
    },
  }));
}

function readInput(input: unknown): { ok: true; messages: ChatMessage[] } | Rejected {
  if (typeof input === "string") {
    return { ok: true, messages: input ? [{ role: "user", content: input }] : [] };
  }
  if (input == null) return { ok: true, messages: [] };
  if (!Array.isArray(input)) return reject(400, "input must be a string or an array");
  const messages: ChatMessage[] = [];
  for (const entry of input) {
    const item = record(entry);
    if (!item) continue;
    const type = typeof item.type === "string" ? item.type : "";
    if (type === "item_reference") return reject(400, "item references are not supported; send the full input");
    if (type === "function_call") {
      const name = stringValue(item.name);
      const callId = stringValue(item.call_id) ?? stringValue(item.id);
      if (!name || !callId) continue;
      const call = {
        id: callId,
        type: "function",
        function: { name, arguments: argumentsText(item.arguments) },
      };
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && Array.isArray(last.tool_calls)) last.tool_calls.push(call);
      else messages.push({ role: "assistant", content: null, tool_calls: [call] });
      continue;
    }
    if (type === "function_call_output") {
      const callId = stringValue(item.call_id) ?? stringValue(item.id);
      if (!callId) continue;
      const output = toolOutput(item.output);
      if (!output.ok) return output;
      messages.push({
        role: "tool",
        tool_call_id: callId,
        ...(stringValue(item.name) ? { name: stringValue(item.name) } : {}),
        content: output.text,
      });
      continue;
    }
    if (type === "reasoning") continue;
    const role = stringValue(item.role);
    if (type === "message" || role) {
      const content = chatContent(item.content);
      if (!content.ok) return content;
      messages.push({ role: role ?? "user", content: content.content });
    }
  }
  return { ok: true, messages };
}

function readTools(value: unknown): { ok: true; tools?: ChatMessage[] } | Rejected {
  if (value == null) return { ok: true };
  if (!Array.isArray(value)) return reject(400, "tools must be an array");
  const tools: ChatMessage[] = [];
  for (const entry of value) {
    const tool = record(entry);
    if (!tool) continue;
    const nested = record(tool.function);
    const type = typeof tool.type === "string" ? tool.type : "";
    if (type !== "function" && !nested) continue;
    const name = stringValue(nested?.name) ?? stringValue(tool.name);
    if (!name) continue;
    const parameters = record(nested?.parameters) ?? record(tool.parameters);
    const description = stringValue(nested?.description) ?? stringValue(tool.description);
    tools.push({
      type: "function",
      function: {
        name,
        ...(description ? { description } : {}),
        ...(parameters ? { parameters } : {}),
      },
    });
  }
  return { ok: true, ...(tools.length > 0 ? { tools } : {}) };
}

function readToolChoice(value: unknown): { ok: true; choice?: unknown } | Rejected {
  if (value == null) return { ok: true };
  if (value === "none" || value === "auto" || value === "required") return { ok: true, choice: value };
  const choice = record(value);
  if (!choice) return reject(400, "tool_choice is invalid");
  const nested = record(choice.function);
  const name = stringValue(nested?.name) ?? stringValue(choice.name);
  if (choice.type === "function" && name) return { ok: true, choice: { type: "function", function: { name } } };
  if (typeof choice.type === "string" && choice.type !== "function") {
    return reject(400, `${choice.type} tool_choice is not supported`);
  }
  return reject(400, "tool_choice is invalid");
}

function chatContent(content: unknown): { ok: true; content: unknown } | Rejected {
  if (typeof content === "string" || content == null) return { ok: true, content: content ?? "" };
  if (!Array.isArray(content)) return { ok: true, content: "" };
  const parts: Array<Record<string, unknown>> = [];
  for (const entry of content) {
    const part = record(entry);
    if (!part) continue;
    const type = part.type;
    if ((type === "input_text" || type === "output_text" || type === "text") && typeof part.text === "string") {
      parts.push({ type: "text", text: part.text });
      continue;
    }
    if (type === "refusal" && typeof part.refusal === "string") {
      parts.push({ type: "text", text: part.refusal });
      continue;
    }
    if (type === "input_image" || type === "image_url") {
      const url = imageUrl(part);
      if (!url) {
        if (typeof part.file_id === "string" && part.file_id.trim()) return reject(400, "image file ids are not supported");
        continue;
      }
      parts.push({ type: "image_url", image_url: { url } });
    }
  }
  return { ok: true, content: parts };
}

function toolOutput(value: unknown): { ok: true; text: string } | Rejected {
  if (typeof value === "string") return { ok: true, text: value };
  if (value == null) return { ok: true, text: "" };
  if (!Array.isArray(value)) return { ok: true, text: JSON.stringify(value) };
  const content = chatContent(value);
  if (!content.ok) return content;
  if (typeof content.content === "string") return { ok: true, text: content.content };
  if (!Array.isArray(content.content) || content.content.length === 0) return { ok: true, text: JSON.stringify(value) };
  return {
    ok: true,
    text: content.content.map((entry) => {
      const part = record(entry);
      return typeof part?.text === "string" ? part.text : "";
    }).filter(Boolean).join(""),
  };
}

function imageUrl(part: Record<string, unknown>): string | undefined {
  if (typeof part.image_url === "string" && part.image_url.trim()) return part.image_url.trim();
  return stringValue(record(part.image_url)?.url);
}

function argumentsText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "{}";
  return JSON.stringify(value);
}

function ensureStarted(state: ResponsesStreamState): Array<Record<string, unknown>> {
  if (state.started) return [];
  state.started = true;
  const response = snapshot(state, "in_progress");
  return [
    { type: "response.created", response },
    { type: "response.in_progress", response },
  ];
}

function appendReasoning(state: ResponsesStreamState, delta: string): Array<Record<string, unknown>> {
  const events = closeKind(state, "message");
  const opened = openPiece(state, "reasoning");
  if (opened.piece.kind !== "reasoning") return events;
  events.push(...opened.events);
  opened.piece.text += delta;
  events.push({
    type: "response.reasoning_summary_text.delta",
    item_id: opened.piece.id,
    output_index: outputIndex(state, opened.piece.id),
    summary_index: 0,
    delta,
  });
  return events;
}

function appendText(state: ResponsesStreamState, delta: string): Array<Record<string, unknown>> {
  const events = closeKind(state, "reasoning");
  const opened = openPiece(state, "message");
  if (opened.piece.kind !== "message") return events;
  events.push(...opened.events);
  opened.piece.text += delta;
  events.push({
    type: "response.output_text.delta",
    item_id: opened.piece.id,
    output_index: outputIndex(state, opened.piece.id),
    content_index: 0,
    delta,
  });
  return events;
}

function appendTool(state: ResponsesStreamState, value: unknown): Array<Record<string, unknown>> {
  const call = record(value);
  if (!call) return [];
  const events = [...closeKind(state, "reasoning"), ...closeKind(state, "message")];
  const fn = record(call.function) ?? {};
  const index = typeof call.index === "number" && Number.isFinite(call.index) ? call.index : state.toolAt.size;
  const name = typeof fn.name === "string" ? fn.name : "";
  const args = typeof fn.arguments === "string" ? fn.arguments : "";
  const callId = stringValue(call.id);
  let position = state.toolAt.get(index);
  if (position === undefined) {
    const id = nextId(state, "fc");
    const piece: Piece = { kind: "tool", id, callId: callId ?? id, name, arguments: "", done: false };
    position = state.pieces.length;
    state.pieces.push(piece);
    state.toolAt.set(index, position);
    events.push({ type: "response.output_item.added", output_index: position, item: toolItem(piece, false) });
  }
  const piece = state.pieces[position];
  if (!piece || piece.kind !== "tool") return events;
  if (callId) piece.callId = callId;
  if (name) piece.name = name;
  if (args) {
    piece.arguments += args;
    events.push({
      type: "response.function_call_arguments.delta",
      item_id: piece.id,
      output_index: position,
      delta: args,
    });
  }
  return events;
}

function openPiece(
  state: ResponsesStreamState,
  kind: "reasoning" | "message",
): { events: Array<Record<string, unknown>>; piece: Piece } {
  const existing = [...state.pieces].reverse().find((piece) => piece.kind === kind && !piece.done);
  if (existing) return { events: [], piece: existing };
  const id = nextId(state, kind === "reasoning" ? "rs" : "msg");
  const index = state.pieces.length;
  if (kind === "reasoning") {
    const piece: Extract<Piece, { kind: "reasoning" }> = { kind: "reasoning", id, text: "", done: false };
    state.pieces.push(piece);
    return {
      piece,
      events: [
        { type: "response.output_item.added", output_index: index, item: reasoningItem(piece, false) },
        {
          type: "response.reasoning_summary_part.added",
          item_id: id,
          output_index: index,
          summary_index: 0,
          part: { type: "summary_text", text: "" },
        },
      ],
    };
  }
  const piece: Extract<Piece, { kind: "message" }> = { kind: "message", id, text: "", done: false };
  state.pieces.push(piece);
  return {
    piece,
    events: [
      { type: "response.output_item.added", output_index: index, item: messageItem(piece, false) },
      {
        type: "response.content_part.added",
        item_id: id,
        output_index: index,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      },
    ],
  };
}

function closeKind(state: ResponsesStreamState, kind: "reasoning" | "message"): Array<Record<string, unknown>> {
  const events: Array<Record<string, unknown>> = [];
  for (const piece of state.pieces) {
    if (piece.kind !== kind || piece.done) continue;
    piece.done = true;
    const index = outputIndex(state, piece.id);
    if (piece.kind === "reasoning") {
      events.push(
        {
          type: "response.reasoning_summary_text.done",
          item_id: piece.id,
          output_index: index,
          summary_index: 0,
          text: piece.text,
        },
        { type: "response.output_item.done", output_index: index, item: reasoningItem(piece, true) },
      );
    } else if (piece.kind === "message") {
      const part = { type: "output_text", text: piece.text, annotations: [] };
      events.push(
        {
          type: "response.output_text.done",
          item_id: piece.id,
          output_index: index,
          content_index: 0,
          text: piece.text,
        },
        { type: "response.content_part.done", item_id: piece.id, output_index: index, content_index: 0, part },
        { type: "response.output_item.done", output_index: index, item: messageItem(piece, true) },
      );
    }
  }
  return events;
}

function closeOpen(state: ResponsesStreamState): Array<Record<string, unknown>> {
  const events = [...closeKind(state, "reasoning"), ...closeKind(state, "message")];
  for (const [position, piece] of state.pieces.entries()) {
    if (piece.kind !== "tool" || piece.done) continue;
    piece.done = true;
    events.push(
      {
        type: "response.function_call_arguments.done",
        item_id: piece.id,
        output_index: position,
        arguments: piece.arguments,
      },
      { type: "response.output_item.done", output_index: position, item: toolItem(piece, true) },
    );
  }
  return events;
}

function applyFinish(state: ResponsesStreamState, reason: string): void {
  if (reason === "length") {
    state.status = "incomplete";
    state.incompleteReason = "max_output_tokens";
    return;
  }
  if (reason === "content_filter") {
    state.status = "incomplete";
    state.incompleteReason = "content_filter";
  }
}

function snapshot(
  state: ResponsesStreamState,
  status: "in_progress" | ResponsesStreamState["status"] = state.status,
): Record<string, unknown> {
  return {
    id: state.id,
    object: "response",
    created_at: state.createdAt,
    status,
    model: state.model,
    output: state.pieces.map((piece) => {
      if (piece.kind === "reasoning") return reasoningItem(piece, piece.done);
      if (piece.kind === "message") return messageItem(piece, piece.done);
      return toolItem(piece, piece.done);
    }),
    ...(state.incompleteReason ? { incomplete_details: { reason: state.incompleteReason } } : {}),
    ...(state.usage ? { usage: state.usage } : {}),
  };
}

function messageItem(piece: Extract<Piece, { kind: "message" }>, done: boolean): Record<string, unknown> {
  return {
    id: piece.id,
    type: "message",
    role: "assistant",
    status: done ? "completed" : "in_progress",
    content: done ? [{ type: "output_text", text: piece.text, annotations: [] }] : [],
  };
}

function reasoningItem(piece: Extract<Piece, { kind: "reasoning" }>, done: boolean): Record<string, unknown> {
  return {
    id: piece.id,
    type: "reasoning",
    status: done ? "completed" : "in_progress",
    summary: done ? [{ type: "summary_text", text: piece.text }] : [],
    content: done ? [{ type: "reasoning_text", text: piece.text }] : [],
  };
}

function toolItem(piece: Extract<Piece, { kind: "tool" }>, done: boolean): Record<string, unknown> {
  return {
    id: piece.id,
    type: "function_call",
    status: done ? "completed" : "in_progress",
    call_id: piece.callId,
    name: piece.name,
    arguments: piece.arguments,
  };
}

function outputIndex(state: ResponsesStreamState, id: string): number {
  return state.pieces.findIndex((piece) => piece.id === id);
}

function responsesUsage(usage: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (!usage) return undefined;
  const input = numberValue(usage.prompt_tokens) ?? numberValue(usage.input_tokens);
  const output = numberValue(usage.completion_tokens) ?? numberValue(usage.output_tokens);
  if (input === undefined && output === undefined) return undefined;
  const inputTokens = input ?? 0;
  const outputTokens = output ?? 0;
  const promptDetails = record(usage.prompt_tokens_details) ?? record(usage.input_tokens_details);
  const completionDetails = record(usage.completion_tokens_details) ?? record(usage.output_tokens_details);
  const cached = numberValue(promptDetails?.cached_tokens);
  const reasoning = numberValue(completionDetails?.reasoning_tokens);
  const total = numberValue(usage.total_tokens) ?? inputTokens + outputTokens;
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: total,
    ...(cached !== undefined && cached > 0 ? { input_tokens_details: { cached_tokens: cached } } : {}),
    ...(reasoning !== undefined && reasoning > 0 ? { output_tokens_details: { reasoning_tokens: reasoning } } : {}),
  };
}

function nextId(state: ResponsesStreamState, prefix: string): string {
  state.seq += 1;
  return `${prefix}_${state.seq}`;
}

function present(value: unknown): boolean {
  if (typeof value === "string") return value.trim().length > 0;
  return value != null && value !== false;
}

function reject(status: number, message: string): Rejected {
  return { ok: false, status, message };
}

function arrayAt(value: unknown, index: number): unknown {
  return Array.isArray(value) ? value[index] : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
