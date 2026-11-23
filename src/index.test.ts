import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import type { PluginLogFields, PluginLogger, SubscriptionInput } from "@viateai/sdk";
import { ANTIGRAVITY_CLIENT_ID } from "./creds.js";
import {
  chatCompletions,
  closeAntigravityLoopback,
  imageEdits,
  imageGenerations,
  listModels,
  normalizeCredentials,
  plugin,
  responses,
  startOAuth,
  subscription,
} from "./index.js";
import {
  ANTIGRAVITY_DEFAULT_SYSTEM,
  buildAntigravityEnvelope,
  createStreamState,
  resolveAntigravityWireModel,
  takeSseEvents,
  toOpenAIChatCompletion,
  toOpenAIStreamChunks,
} from "./translate.js";

const originalFetch = globalThis.fetch;

afterEach(async () => {
  globalThis.fetch = originalFetch;
  await closeAntigravityLoopback();
});

describe("antigravity translation", () => {
  it("keeps the requested model and moves client system text onto the first user turn", () => {
    const envelope = buildAntigravityEnvelope({
      model: "antigravity/gemini-3.8-flash-tiered",
      messages: [
        { role: "system", content: "Be brief" },
        { role: "user", content: "hi" },
      ],
    }, "aicode-consumers", "user@example.com");
    assert.equal(envelope.model, "gemini-3.8-flash-tiered");
    assert.equal(envelope.project, "aicode-consumers");
    assert.equal(envelope.request.systemInstruction.parts[0]?.text, ANTIGRAVITY_DEFAULT_SYSTEM);
    assert.deepEqual(envelope.request.contents[0]?.parts.slice(0, 2), [
      { text: "Be brief" },
      { text: "hi" },
    ]);
    assert.equal(envelope.request.generationConfig.maxOutputTokens, undefined);
  });

  it("sends tool calls with the Antigravity thought-signature bypass", () => {
    const envelope = buildAntigravityEnvelope({
      model: "gemini-3.8-flash-tiered",
      messages: [
        { role: "user", content: "run" },
        {
          role: "assistant",
          tool_calls: [{
            id: "call_1",
            type: "function",
            function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" },
          }],
        },
        { role: "tool", tool_call_id: "call_1", content: "ok" },
      ],
      tools: [{ type: "function", function: { name: "read_file", parameters: { type: "object" } } }],
    }, "project", "user@example.com");
    const modelTurn = envelope.request.contents.find((entry) => entry.role === "model");
    assert.equal(modelTurn?.parts[0]?.thoughtSignature, "skip_thought_signature_validator");
    assert.deepEqual(modelTurn?.parts[0]?.functionCall, {
      id: "call_1",
      name: "read_file",
      args: { path: "a.ts" },
    });
    assert.equal(envelope.request.toolConfig && typeof envelope.request.toolConfig, "object");
  });

  it("rewrites JSON Schema tool parameters into the Gemini schema subset", () => {
    const envelope = buildAntigravityEnvelope({
      model: "gemini-4-argon-medium",
      messages: [{ role: "user", content: "hi" }],
      tools: [{
        type: "function",
        function: {
          name: "create_workspace",
          parameters: {
            $schema: "https://json-schema.org/draft/2020-12/schema",
            type: "object",
            additionalProperties: false,
            properties: {
              isolation: { type: "string", enum: ["local", "worktree"], minLength: 1 },
              prNumber: { type: "integer", exclusiveMinimum: 0, maximum: 10 },
              skip: { type: ["number", "null"] },
              labels: {
                type: "object",
                propertyNames: { type: "string" },
                additionalProperties: { type: "string" },
              },
              response: {
                anyOf: [
                  { type: "object", properties: { behavior: { type: "string", const: "allow" } } },
                ],
              },
            },
            required: ["prNumber"],
          },
        },
      }],
    }, "project", "user@example.com");
    const tools = envelope.request.tools?.[0] as { functionDeclarations: Array<{ parameters: unknown }> };
    assert.deepEqual(tools.functionDeclarations[0]?.parameters, {
      type: "object",
      additionalProperties: false,
      properties: {
        isolation: { type: "string", enum: ["local", "worktree"], minLength: 1 },
        prNumber: { type: "integer", minimum: 0, maximum: 10 },
        skip: { type: "number", nullable: true },
        labels: { type: "object", additionalProperties: { type: "string" } },
        response: {
          anyOf: [
            { type: "object", properties: { behavior: { type: "string", enum: ["allow"] } } },
          ],
        },
      },
      required: ["prNumber"],
    });
  });

  it("flattens Claude tool schemas Vertex rejects as draft 2020-12", () => {
    const envelope = buildAntigravityEnvelope({
      model: "claude-opus-5-5",
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          type: "function",
          function: {
            name: "todo",
            parameters: {
              type: "object",
              additionalProperties: false,
              properties: {
                task: {
                  description: "verbatim task content; array targets several (not start)",
                  anyOf: [
                    { type: "string" },
                    { type: "array", items: { type: "string" } },
                  ],
                },
                items: { type: "array", items: { type: "string" }, minItems: 1 },
                title: { type: "string", minLength: 1, default: "Tasks" },
              },
              required: ["task"],
            },
          },
        },
        {
          type: "function",
          function: {
            name: "respond_to_permission",
            parameters: {
              type: "object",
              properties: {
                response: {
                  anyOf: [
                    {
                      type: "object",
                      additionalProperties: false,
                      properties: { behavior: { type: "string", const: "allow" } },
                      required: ["behavior"],
                    },
                    {
                      type: "object",
                      additionalProperties: false,
                      properties: {
                        behavior: { type: "string", const: "deny" },
                        message: { type: "string" },
                      },
                      required: ["behavior"],
                    },
                  ],
                },
              },
              required: ["response"],
            },
          },
        },
      ],
    }, "project", "user@example.com");
    const tools = envelope.request.tools?.[0] as { functionDeclarations: Array<{ name: string; parameters: unknown }> };
    assert.deepEqual(tools.functionDeclarations[0]?.parameters, {
      type: "object",
      additionalProperties: false,
      properties: {
        task: {
          description: "verbatim task content; array targets several (not start)",
          type: "string",
        },
        items: { type: "array", items: { type: "string" } },
        title: { type: "string" },
      },
      required: ["task"],
    });
    assert.deepEqual(tools.functionDeclarations[1]?.parameters, {
      type: "object",
      properties: {
        response: {
          type: "object",
          additionalProperties: false,
          properties: {
            behavior: { type: "string", enum: ["allow", "deny"] },
            message: { type: "string" },
          },
          required: ["behavior"],
        },
      },
      required: ["response"],
    });
  });

  it("maps thoughts, text, and usage into an OpenAI completion", () => {
    const completion = toOpenAIChatCompletion("gemini-3.8-flash-tiered", {
      response: {
        candidates: [{
          finishReason: "STOP",
          content: { parts: [{ thought: true, text: "think" }, { text: "answer" }] },
        }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2, thoughtsTokenCount: 4, totalTokenCount: 9 },
      },
    });
    const message = (completion.choices as Array<{ message: Record<string, unknown> }>)[0]?.message;
    assert.equal(message?.content, "answer");
    assert.equal(message?.reasoning_content, "think");
    assert.deepEqual(completion.usage, {
      prompt_tokens: 3,
      completion_tokens: 6,
      total_tokens: 9,
      completion_tokens_details: { reasoning_tokens: 4 },
    });
  });

  it("turns upstream SSE into OpenAI chunks", () => {
    const state = createStreamState();
    const taken = takeSseEvents([
      `data: ${JSON.stringify({
        response: {
          candidates: [{ content: { parts: [{ thought: true, text: "a" }, { text: "b" }] }, finishReason: "STOP" }],
          usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, thoughtsTokenCount: 1 },
        },
      })}`,
      "",
      "",
    ].join("\n"));
    const frames = taken.events.flatMap((event) => toOpenAIStreamChunks("gemini-3.8-flash-tiered", event, state));
    assert.equal(frames[0]?.choices && (frames[0].choices as Array<{ delta: { reasoning_content?: string } }>)[0]?.delta.reasoning_content, "a");
    assert.equal((frames[1]?.choices as Array<{ delta: { content?: string } }>)[0]?.delta.content, "b");
    assert.equal(frames.at(-1)?.usage && (frames.at(-1)?.usage as { completion_tokens: number }).completion_tokens, 2);
  });
});

describe("antigravity provider", () => {
  it("normalizes a pasted token payload", () => {
    const normalized = normalizeCredentials({
      access_token: "access",
      refresh_token: "refresh",
      expires_in: 3600,
      email: "user@example.com",
      providerSpecificData: { projectId: "aicode-consumers" },
    });
    const credentials = normalized.credentials as { projectId: string; refreshToken: string; expiresAt: string };
    assert.equal(credentials.projectId, "aicode-consumers");
    assert.equal(credentials.refreshToken, "refresh");
    assert.equal(normalized.displayLabel, "user@example.com");
    assert.ok(Date.parse(credentials.expiresAt) > Date.now());
  });

  it("opens a loopback callback and points Google at it", async () => {
    const started = await startOAuth({
      connectionId: "c1",
      credentials: {},
      callbackUrl: "http://127.0.0.1:4317/api/admin/oauth/callback?state=state-1",
    });
    assert.equal(started.type, "authorization_url");
    if (started.type !== "authorization_url") return;
    const url = new URL(started.authorizationUrl);
    assert.equal(url.origin, "https://accounts.google.com");
    assert.equal(url.searchParams.get("client_id"), ANTIGRAVITY_CLIENT_ID);
    assert.match(ANTIGRAVITY_CLIENT_ID, /\.apps\.googleusercontent\.com$/);
    assert.equal(url.searchParams.get("state"), "state-1");
    const redirect = new URL(url.searchParams.get("redirect_uri") ?? "");
    assert.equal(redirect.hostname, "127.0.0.1");
    assert.equal(redirect.pathname, "/callback");
  });

  it("posts chat to Cloud Code and falls over after a 503", async () => {
    const urls: string[] = [];
    globalThis.fetch = async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.includes("daily-cloudcode-pa")) return new Response("busy", { status: 503 });
      return Response.json({
        response: {
          candidates: [{ finishReason: "STOP", content: { parts: [{ text: "ok" }] } }],
          usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
        },
      });
    };
    const result = await chatCompletions(
      { body: { model: "gemini-3.8-flash-tiered", messages: [{ role: "user", content: "hi" }] }, headers: {} },
      connection(),
    );
    assert.equal(result.status, 200);
    assert.equal(urls.length, 2);
    assert.match(urls[1] ?? "", /cloudcode-pa\.googleapis\.com\/v1internal:generateContent$/);
    const body = JSON.parse(await new Response(result.body).text()) as { choices: Array<{ message: { content: string } }> };
    assert.equal(body.choices[0]?.message.content, "ok");
  });

  it("lists live model ids and skips internal ones", async () => {
    let userAgent = "";
    globalThis.fetch = async (_input, init) => {
      userAgent = new Headers(init?.headers).get("user-agent") ?? "";
      return Response.json({
        models: {
          "gemini-3.8-flash-tiered": { displayName: "Gemini 3.8 Flash" },
          "internal-model": { displayName: "Hidden", isInternal: true },
        },
      });
    };
    const listed = await listModels({ headers: {} }, connection());
    assert.deepEqual(listed.data.map((model) => model.id), [
      "gemini-3.8-flash-tiered",
      "gemini-3.1-flash-image",
    ]);
    assert.equal(listed.data[0]?.name, "Gemini 3.8 Flash");
    assert.equal(userAgent, "antigravity/hub/2.29.0 darwin/arm64");
  });

  it("collapses thinking-level suffixes and sends the spliced id upstream", async () => {
    let seenModel = "";
    globalThis.fetch = async (input, init) => {
      const url = String(input);
      if (url.includes("fetchAvailableModels") || init?.body === "{}") {
        return Response.json({
          models: {
            "gemini-3.8-flash-low": { displayName: "Gemini 3.8 Flash (Low)" },
            "gemini-3.8-flash-high": { displayName: "Gemini 3.8 Flash (High)" },
            "gemini-4-argon-high": { displayName: "Gemini 4 Argon (High)" },
            "gemini-3.8-flash-tiered": { displayName: "Gemini 3.8 Flash" },
            "claude-opus-4-6-thinking": { displayName: "Claude Opus 4.6 Thinking" },
            "gpt-oss-120b-medium": { displayName: "gpt-oss 120b medium" },
          },
        });
      }
      seenModel = String(record(JSON.parse(String(init?.body))).model);
      return Response.json({
        response: {
          candidates: [{ finishReason: "STOP", content: { parts: [{ text: "ok" }] } }],
        },
      });
    };
    const listed = await listModels({ headers: {} }, connection());
    assert.deepEqual(listed.data.map((model) => model.id), [
      "gemini-3.8-flash",
      "gemini-4-argon",
      "gemini-3.8-flash-tiered",
      "claude-opus-4-6-thinking",
      "gpt-oss-120b-medium",
      "gemini-3.1-flash-image",
    ]);
    assert.equal(listed.data[0]?.name, "Gemini 3.8 Flash");
    assert.equal(resolveAntigravityWireModel("gemini-3.8-flash", "high"), "gemini-3.8-flash-high");
    assert.equal(resolveAntigravityWireModel("gemini-4-argon"), "gemini-4-argon-high");
    assert.equal(resolveAntigravityWireModel("gemini-3.8-flash-tiered", "high"), "gemini-3.8-flash-tiered");
    assert.equal(resolveAntigravityWireModel("gpt-oss-120b-medium", "low"), "gpt-oss-120b-medium");
    const result = await chatCompletions(
      { body: { model: "gemini-3.8-flash", reasoning_effort: "high", messages: [{ role: "user", content: "hi" }] }, headers: {} },
      connection(),
    );
    assert.equal(result.status, 200);
    assert.equal(result.upstreamModel, "gemini-3.8-flash-high");
    assert.equal(seenModel, "gemini-3.8-flash-high");
    const completion = JSON.parse(await new Response(result.body).text()) as { model?: string };
    assert.equal(completion.model, "gemini-3.8-flash");
    const responded = await responses(
      { body: { model: "gemini-3.8-flash", input: "hi", reasoning: { effort: "low" } }, headers: {} },
      connection(),
    );
    assert.equal(responded.status, 200);
    assert.equal(responded.upstreamModel, "gemini-3.8-flash-low");
    assert.equal(seenModel, "gemini-3.8-flash-low");
  });

  it("generates Nano Banana images through Cloud Code", async () => {
    let seenBody: unknown;
    globalThis.fetch = async (input, init) => {
      assert.match(String(input), /\/v1internal:generateContent$/);
      seenBody = JSON.parse(String(init?.body));
      return Response.json({
        response: {
          candidates: [{
            content: {
              parts: [
                { text: "a horse" },
                { inlineData: { mimeType: "image/png", data: "aW1n" } },
              ],
            },
          }],
        },
      });
    };
    const result = await imageGenerations(
      { body: { model: "gemini-3.1-flash-image", prompt: "a horse", size: "1792x1024", n: 1 }, headers: {} },
      connection(),
    );
    assert.equal(result.status, 200);
    assert.equal(result.upstreamModel, "gemini-3.1-flash-image");
    const envelope = record(seenBody);
    assert.equal(envelope.requestType, "image_gen");
    assert.equal(envelope.model, "gemini-3.1-flash-image");
    const request = record(envelope.request);
    const generation = record(request.generationConfig);
    assert.equal(record(generation.imageConfig).aspectRatio, "16:9");
    const payload = JSON.parse(await new Response(result.body).text()) as { data: Array<{ b64_json: string }> };
    assert.equal(payload.data[0]?.b64_json, "aW1n");
  });

  it("sends edit references as inline image parts", async () => {
    let seenBody: unknown;
    globalThis.fetch = async (_input, init) => {
      seenBody = JSON.parse(String(init?.body));
      return Response.json({
        response: { candidates: [{ content: { parts: [{ inlineData: { data: "ZWRpdA" } }] } }] },
      });
    };
    const result = await imageEdits(
      {
        body: {
          model: "gemini-3.1-flash-image",
          prompt: "make it blue",
          images: [{ b64_json: "cG5n", media_type: "image/png" }],
        },
        headers: {},
      },
      connection(),
    );
    assert.equal(result.status, 200);
    const request = record(record(seenBody).request);
    const contents = request.contents;
    assert.ok(Array.isArray(contents));
    const first = record(contents[0]);
    const parts = first.parts;
    assert.ok(Array.isArray(parts));
    assert.deepEqual(record(parts[0]).inlineData, { mimeType: "image/png", data: "cG5n" });
  });
});

describe("antigravity responses", () => {
  it("registers a responses handler", () => {
    let registered: { responses?: unknown } | undefined;
    plugin.register({
      registerProvider(handler) {
        registered = handler;
      },
      log: quietLog(),
    });
    assert.equal(typeof registered?.responses, "function");
  });

  it("translates a responses request into Cloud Code and returns a response object", async () => {
    let seenBody: unknown;
    globalThis.fetch = async (_input, init) => {
      seenBody = JSON.parse(String(init?.body));
      return Response.json({
        response: {
          candidates: [{
            finishReason: "STOP",
            content: { parts: [{ thought: true, text: "think" }, { text: "answer" }, { functionCall: { id: "call_1", name: "read_file", args: { path: "a.ts" } } }] },
          }],
          usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 2, thoughtsTokenCount: 1, totalTokenCount: 7, cachedContentTokenCount: 3 },
        },
      });
    };
    const result = await responses({
      body: {
        model: "antigravity/gemini-3.8-flash-tiered",
        instructions: "Be brief",
        input: "hi",
        max_output_tokens: 32,
        tool_choice: "required",
        tools: [{ type: "function", name: "read_file", parameters: { type: "object" } }],
      },
      headers: {},
      signal: AbortSignal.timeout(5_000),
    }, connection());
    assert.equal(result.status, 200);
    assert.equal(result.parseUsage?.name, "parseOpenAIResponsesUsage");
    const envelope = record(seenBody);
    const request = record(envelope.request);
    const contents = request.contents;
    assert.ok(Array.isArray(contents));
    const first = record(contents[0]);
    const parts = first.parts;
    assert.ok(Array.isArray(parts));
    assert.equal(record(parts[0]).text, "Be brief");
    assert.equal(record(parts[1]).text, "hi");
    assert.equal(record(request.generationConfig).maxOutputTokens, 32);
    assert.equal(record(record(request.toolConfig).functionCallingConfig).mode, "ANY");
    const body = JSON.parse(await new Response(result.body).text()) as {
      object: string;
      status: string;
      output: Array<{ type: string; text?: string; name?: string; content?: Array<{ text?: string }> }>;
      usage: { input_tokens: number; output_tokens: number; input_tokens_details?: { cached_tokens: number } };
    };
    assert.equal(body.object, "response");
    assert.equal(body.status, "completed");
    assert.equal(body.output.find((item) => item.type === "message")?.content?.[0]?.text, "answer");
    assert.equal(body.output.find((item) => item.type === "function_call")?.name, "read_file");
    assert.deepEqual(body.usage, {
      input_tokens: 4,
      output_tokens: 3,
      total_tokens: 7,
      input_tokens_details: { cached_tokens: 3 },
      output_tokens_details: { reasoning_tokens: 1 },
    });
    assert.deepEqual(result.parseUsage?.([body]), {
      inputTokens: 4,
      outputTokens: 3,
      cacheReadTokens: 3,
      cacheWriteTokens: 0,
    });
  });

  it("streams responses events and puts usage on response.completed", async () => {
    globalThis.fetch = async (input) => {
      assert.match(String(input), /streamGenerateContent/);
      return new Response([
        `data: ${JSON.stringify({
          response: {
            candidates: [{ content: { parts: [{ text: "hi" }] }, finishReason: "STOP" }],
            usageMetadata: { promptTokenCount: 2, candidatesTokenCount: 1, totalTokenCount: 3 },
          },
        })}`,
        "",
        "",
      ].join("\n"), { headers: { "content-type": "text/event-stream" } });
    };
    const result = await responses({
      body: { model: "gemini-3.8-flash-tiered", input: "hi", stream: true },
      headers: {},
      signal: AbortSignal.timeout(5_000),
    }, connection());
    const text = await new Response(result.body).text();
    const events = text.split("\n\n").flatMap((block) => {
      const data = block.split("\n").find((line) => line.startsWith("data:"));
      return data ? [JSON.parse(data.slice(5)) as { type?: string; delta?: string; response?: { usage?: unknown } }] : [];
    });
    assert.deepEqual(events.map((event) => event.type), [
      "response.created",
      "response.in_progress",
      "response.output_item.added",
      "response.content_part.added",
      "response.output_text.delta",
      "response.output_text.done",
      "response.content_part.done",
      "response.output_item.done",
      "response.completed",
    ]);
    assert.equal(events.find((event) => event.type === "response.output_text.delta")?.delta, "hi");
    const completed = events.find((event) => event.type === "response.completed");
    assert.deepEqual(result.parseUsage?.([completed]), {
      inputTokens: 2,
      outputTokens: 1,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
    });
  });

  it("rejects stored response ids without calling upstream", async () => {
    globalThis.fetch = async () => {
      throw new Error("responses should not call upstream");
    };
    const result = await responses({
      body: { model: "gemini-3.8-flash-tiered", previous_response_id: "resp_prev", input: "hi" },
      headers: {},
      signal: AbortSignal.timeout(5_000),
    }, connection());
    assert.equal(result.status, 400);
    const body = JSON.parse(await new Response(result.body).text()) as { error: { message: string } };
    assert.match(body.error.message, /previous_response_id/);
  });
});

describe("antigravity subscription", () => {
  it("registers a subscription handler", () => {
    let registered: { subscription?: unknown } | undefined;
    plugin.register({
      registerProvider(handler) {
        registered = handler;
      },
      log: quietLog(),
    });
    assert.equal(typeof registered?.subscription, "function");
  });

  it("maps live quota, weekly families, and catalog buckets that are not stale-full", async () => {
    const bodies: unknown[] = [];
    globalThis.fetch = async (input, init) => {
      const url = String(input);
      bodies.push(JSON.parse(String(init?.body)));
      assert.equal(new Headers(init?.headers).get("authorization"), "Bearer access");
      if (url.endsWith(":retrieveUserQuotaSummary")) {
        return Response.json({
          quotaSummary: {
            groups: [
              {
                displayName: "Gemini Models",
                buckets: [
                  { bucketId: "gemini-5h", displayName: "5h", remainingFraction: 0.5 },
                  { bucketId: "gemini-weekly", displayName: "Weekly", remainingFraction: 0.4, resetTime: "2026-11-14T00:00:00Z" },
                ],
              },
              {
                displayName: "Claude and GPT models",
                buckets: [{ displayName: "Weekly", remainingFraction: 1, disabled: true }],
              },
            ],
          },
        });
      }
      if (url.endsWith(":retrieveUserQuota")) {
        return Response.json({
          buckets: [
            { modelId: "models/gemini-3.8-flash-tiered", remainingFraction: 0.25, resetTime: "2026-11-07T12:00:00Z" },
            { modelId: "hidden-model", remainingFraction: 0.1, resetTime: "2026-11-07T12:00:00Z" },
            { modelId: "claude-opus", remainingFraction: 0 },
          ],
        });
      }
      if (url.includes("fetchAvailableModels")) {
        return Response.json({
          models: {
            "gemini-3.8-flash-tiered": { displayName: "Gemini 3.8 Flash", quotaInfo: { remainingFraction: 1, resetTime: "2026-11-07T12:00:00Z" } },
            "hidden-model": { displayName: "Hidden", isInternal: true, quotaInfo: { remainingFraction: 0.1 } },
            "stale-full": { displayName: "Stale", quotaInfo: { remainingFraction: 1, resetTime: "2026-11-07T12:00:00Z" } },
            "partial-only": { displayName: "Partial", quotaInfo: { remainingFraction: 0.5, resetTime: "2026-11-07T18:00:00Z" } },
            "tab-complete": { displayName: "Tab", quotaInfo: { remainingFraction: 1 } },
          },
        });
      }
      return new Response("missing", { status: 404 });
    };
    const result = await subscription(subscriptionInput({
      accessToken: "access",
      refreshToken: "refresh",
      projectId: "aicode-consumers",
      tier: "g1-ultra-tier",
    }));
    assert.equal(result.status, "ok");
    if (result.status !== "ok") return;
    assert.equal(result.plan, "Ultra");
    assert.equal(result.nextDueMs, 5 * 60 * 1000);
    assert.ok(bodies.every((body) => record(body).project === "aicode-consumers"));
    assert.deepEqual(result.usages, [
      {
        type: "gemini-3.8-flash-tiered",
        label: "Gemini 3.8 Flash",
        group: "Models",
        unit: "percent",
        used: 75,
        limit: 100,
        remaining: 25,
        resetsAt: "2026-11-07T12:00:00.000Z",
        reported: true,
      },
      {
        type: "claude-opus",
        label: "claude-opus",
        group: "Models",
        unit: "percent",
        used: 100,
        limit: 100,
        remaining: 0,
        reported: true,
      },
      {
        type: "partial-only",
        label: "Partial",
        group: "Models",
        unit: "percent",
        used: 50,
        limit: 100,
        remaining: 50,
        resetsAt: "2026-11-07T18:00:00.000Z",
        reported: true,
      },
      {
        type: "tab-complete",
        label: "Tab",
        group: "Models",
        unit: "percent",
        used: 0,
        limit: null,
        remaining: null,
        unlimited: true,
        reported: true,
      },
      {
        type: "weekly:gemini",
        label: "Gemini Models",
        group: "Weekly",
        unit: "percent",
        used: 60,
        limit: 100,
        remaining: 40,
        resetsAt: "2026-11-14T00:00:00.000Z",
        reported: true,
      },
    ]);
  });

  it("fails over quota hosts and still reads a top-level weekly summary", async () => {
    const urls: string[] = [];
    globalThis.fetch = async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith(":retrieveUserQuota")) {
        if (url.includes("daily-cloudcode-pa")) return new Response("busy", { status: 503 });
        return Response.json({
          buckets: [{ modelId: "gemini-3.8-flash-tiered", remainingFraction: "0.5", resetTime: 1_778_000_000 }],
        });
      }
      if (url.endsWith(":retrieveUserQuotaSummary")) {
        if (url.includes("daily-cloudcode-pa")) return new Response("busy", { status: 500 });
        return Response.json({
          groups: [{ displayName: "Claude and GPT models", buckets: [{ bucketId: "weekly", remainingFraction: 0.2 }] }],
        });
      }
      return new Response("missing", { status: 404 });
    };
    const result = await subscription(subscriptionInput({ accessToken: "access", projectId: "project" }));
    assert.equal(result.status, "ok");
    if (result.status !== "ok") return;
    assert.equal(result.plan, "Antigravity");
    assert.deepEqual(urls.filter((url) => url.endsWith(":retrieveUserQuota")), [
      "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
      "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
    ]);
    assert.deepEqual(result.usages?.map((usage) => usage.type), ["gemini-3.8-flash-tiered", "weekly:claude-gpt"]);
    assert.equal(result.usages?.[0]?.used, 50);
    assert.equal(result.usages?.[0]?.resetsAt, new Date(1_778_000_000 * 1000).toISOString());
    assert.equal(result.usages?.[1]?.remaining, 20);
  });

  it("waits for keepalive when the stored token is expired but refreshable", async () => {
    globalThis.fetch = async () => {
      throw new Error("quota should not be fetched");
    };
    const warnings: string[] = [];
    const result = await subscription(subscriptionInput({
      accessToken: "access",
      refreshToken: "refresh",
      projectId: "project",
      expiresAt: "2000-01-01T00:00:00.000Z",
    }, warnings));
    assert.deepEqual(result, { status: "failed", error: "Antigravity access token expired; waiting for refresh" });
    assert.deepEqual(warnings, ["subscription refresh failed"]);
  });

  it("asks for reauth when the token is expired and cannot be refreshed", async () => {
    const result = await subscription(subscriptionInput({
      accessToken: "access",
      projectId: "project",
      expiresAt: "2000-01-01T00:00:00.000Z",
    }));
    assert.deepEqual(result, { status: "needs_reauth", error: "Antigravity access token expired" });
  });

  it("does not fetch the summary after the quota endpoint rejects the token", async () => {
    const urls: string[] = [];
    globalThis.fetch = async (input) => {
      urls.push(String(input));
      return new Response("denied", { status: 401 });
    };
    const refreshable = await subscription(subscriptionInput({
      accessToken: "access",
      refreshToken: "refresh",
      projectId: "project",
    }));
    const expired = await subscription(subscriptionInput({ accessToken: "access", projectId: "project" }));
    assert.deepEqual(refreshable, { status: "failed", error: "Antigravity access token was rejected" });
    assert.deepEqual(expired, { status: "needs_reauth", error: "Antigravity session expired" });
    assert.deepEqual(urls, [
      "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
      "https://daily-cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota",
    ]);
  });

  it("reports a forbidden quota endpoint without asking for reauth", async () => {
    globalThis.fetch = async () => new Response("forbidden", { status: 403 });
    const result = await subscription(subscriptionInput({ accessToken: "access", projectId: "project" }));
    assert.deepEqual(result, { status: "failed", error: "Antigravity subscription unavailable" });
  });

  it("does not discover a project while refreshing usage", async () => {
    globalThis.fetch = async () => {
      throw new Error("quota should not be fetched");
    };
    const result = await subscription(subscriptionInput({ accessToken: "access" }));
    assert.deepEqual(result, {
      status: "failed",
      error: "Antigravity connection is missing projectId; reconnect OAuth",
    });
  });
});

function connection() {
  return {
    connectionId: "c1",
    providerId: "antigravity",
    credentials: { accessToken: "access", projectId: "aicode-consumers", email: "user@example.com" },
  };
}

function record(value: unknown): Record<string, unknown> {
  assert.ok(value && typeof value === "object" && !Array.isArray(value));
  return value as Record<string, unknown>;
}

function subscriptionInput(credentials: unknown, warnings: string[] = []): SubscriptionInput {
  return { connectionId: "c1", credentials, signal: AbortSignal.timeout(5_000), log: quietLog(warnings) };
}

function quietLog(warnings: string[] = []): PluginLogger {
  const log = {
    debug() {},
    info() {},
    warn(fieldsOrMessage: PluginLogFields | string, message?: string) {
      warnings.push(typeof fieldsOrMessage === "string" ? fieldsOrMessage : message ?? "");
    },
    error() {},
    child() {
      return log;
    },
  };
  return log as PluginLogger;
}
