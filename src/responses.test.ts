import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  chatCompletionToResponse,
  consumeChatChunks,
  createResponsesStream,
  finishResponsesStream,
  responsesRequestToChat,
} from "./responses.js";

describe("responses request mapping", () => {
  it("turns function-call history, images, and flat tools into a chat request", () => {
    const converted = responsesRequestToChat({
      model: "gemini-3.8-flash-tiered",
      instructions: "Be brief",
      tool_choice: { type: "function", name: "read_file" },
      tools: [
        { type: "function", name: "read_file", description: "Read", parameters: { type: "object" } },
        { type: "web_search" },
      ],
      input: [
        { type: "message", role: "user", content: [{ type: "input_text", text: "look" }, { type: "input_image", image_url: "data:image/png;base64,YQ" }] },
        { type: "function_call", call_id: "call_1", name: "read_file", arguments: { path: "a.ts" } },
        { type: "function_call_output", call_id: "call_1", output: [{ type: "input_text", text: "file" }] },
        { type: "reasoning", summary: [] },
      ],
    });
    assert.equal(converted.ok, true);
    if (!converted.ok) return;
    assert.deepEqual(converted.chat.messages, [
      { role: "system", content: "Be brief" },
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          { type: "image_url", image_url: { url: "data:image/png;base64,YQ" } },
        ],
      },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "file" },
    ]);
    assert.deepEqual(converted.chat.tools, [{
      type: "function",
      function: { name: "read_file", description: "Read", parameters: { type: "object" } },
    }]);
    assert.deepEqual(converted.chat.tool_choice, { type: "function", function: { name: "read_file" } });
  });

  it("rejects server-side response state and hosted tool choices", () => {
    assert.equal(responsesRequestToChat({ model: "m", conversation: "conv_1", input: "hi" }).ok, false);
    assert.equal(responsesRequestToChat({ model: "m", background: true, input: "hi" }).ok, false);
    const stored = responsesRequestToChat({ model: "m", store: true, input: "hi" });
    assert.equal(stored.ok, false);
    if (!stored.ok) assert.match(stored.message, /store/);
    assert.equal(responsesRequestToChat({ model: "m", store: false, input: "hi" }).ok, true);
    assert.equal(responsesRequestToChat({ model: "m", input: [{ type: "item_reference", id: "msg_1" }] }).ok, false);
    assert.equal(responsesRequestToChat({ model: "m", input: "hi", tool_choice: { type: "web_search" } }).ok, false);
    const image = responsesRequestToChat({
      model: "m",
      input: [{ type: "message", role: "user", content: [{ type: "input_image", file_id: "file_1" }] }],
    });
    assert.equal(image.ok, false);
    if (image.ok) return;
    assert.match(image.message, /file ids/);
  });
});

describe("responses output", () => {
  it("builds one completed response with reasoning, text, a function call, and usage", () => {
    const response = chatCompletionToResponse("gemini-3.8-flash-tiered", {
      choices: [{
        finish_reason: "tool_calls",
        message: {
          reasoning_content: "think",
          content: "answer",
          tool_calls: [{ id: "call_1", type: "function", function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" } }],
        },
      }],
      usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7, completion_tokens_details: { reasoning_tokens: 1 } },
    });
    assert.equal(response.status, "completed");
    assert.deepEqual(response.output, [
      {
        id: "rs_2",
        type: "reasoning",
        status: "completed",
        summary: [{ type: "summary_text", text: "think" }],
        content: [{ type: "reasoning_text", text: "think" }],
      },
      {
        id: "msg_3",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "answer", annotations: [] }],
      },
      {
        id: "fc_4",
        type: "function_call",
        status: "completed",
        call_id: "call_1",
        name: "read_file",
        arguments: "{\"path\":\"a.ts\"}",
      },
    ]);
    assert.deepEqual(response.usage, {
      input_tokens: 4,
      output_tokens: 3,
      total_tokens: 7,
      output_tokens_details: { reasoning_tokens: 1 },
    });
  });

  it("emits function-call deltas and an incomplete terminal event when the chat stream stops at length", () => {
    const state = createResponsesStream("gemini-3.8-flash-tiered");
    const events = [
      ...consumeChatChunks([
        { choices: [{ delta: { tool_calls: [{ index: 0, id: "call_1", function: { name: "read_file", arguments: "{\"path\":" } }] }, finish_reason: null }] },
        { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "\"a.ts\"}" } }] }, finish_reason: "length" }] },
      ], state),
      ...finishResponsesStream(state),
    ];
    assert.deepEqual(events.map((event) => event.type), [
      "response.created",
      "response.in_progress",
      "response.output_item.added",
      "response.function_call_arguments.delta",
      "response.function_call_arguments.delta",
      "response.function_call_arguments.done",
      "response.output_item.done",
      "response.incomplete",
    ]);
    const done = events.find((event) => event.type === "response.function_call_arguments.done");
    assert.equal(done?.arguments, "{\"path\":\"a.ts\"}");
    const incomplete = record(events.at(-1)?.response);
    assert.equal(incomplete.status, "incomplete");
    assert.deepEqual(incomplete.incomplete_details, { reason: "max_output_tokens" });
  });
});

function record(value: unknown): Record<string, unknown> {
  assert.ok(value && typeof value === "object" && !Array.isArray(value));
  return value as Record<string, unknown>;
}
