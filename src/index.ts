import { createServer, type Server } from "node:http";
import { parseOpenAIChatUsage, parseOpenAIResponsesUsage, type CredentialsWrite, type HttpExchange, type JsonGetRequest, type JsonPostRequest, type KeepaliveInput, type KeepaliveResult, type ModelList, type OAuthExchangeInput, type OAuthStartContext, type OAuthStartResult, type PluginContext, type ProviderConnection, type SubscriptionInput, type SubscriptionResult, type ViatePlugin } from "@viateai/sdk";
import { ANTIGRAVITY_CLIENT_ID, ANTIGRAVITY_CLIENT_SECRET } from "./creds.js";
import { antigravityPlan, antigravityUsages } from "./quota.js";
import { chatCompletionToResponse, responsesRequestToChat, streamResponses } from "./responses.js";
import { antigravityUrl, buildAntigravityEnvelope, collapseAntigravityModels, createStreamState, encodeSse, takeSseEvents, toOpenAIChatCompletion, toOpenAIStreamChunks } from "./translate.js";

const PROVIDER_ID = "antigravity";
const HUB_VERSION = "2.29.0";
const AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL = "https://oauth2.googleapis.com/token";
const USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json";
const SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
];
const RUNTIME_BASES = [
  "https://daily-cloudcode-pa.googleapis.com",
  "https://cloudcode-pa.googleapis.com",
];
const LOAD_CODE_ASSIST_URLS = ["https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"];
const MODEL_URLS = [
  "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
  "https://cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
  "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:fetchAvailableModels",
];
const QUOTA_URLS = RUNTIME_BASES.map((base) => `${base}/v1internal:retrieveUserQuota`);
const QUOTA_SUMMARY_URLS = RUNTIME_BASES.map((base) => `${base}/v1internal:retrieveUserQuotaSummary`);
const SUBSCRIPTION_DUE_MS = 5 * 60 * 1000;
const LOOPBACK_HOST = "127.0.0.1";
const LOOPBACK_PATH = "/callback";
const LOOPBACK_TIMEOUT_MS = 5 * 60 * 1000;
const REFRESH_SKEW_MS = 60_000;

export type AntigravityCredentials = {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: string;
  email?: string;
  projectId?: string;
  tier?: string;
};

type Loopback = { server: Server; timeout: NodeJS.Timeout };

let loopback: Loopback | undefined;

export const plugin: ViatePlugin = {
  id: PROVIDER_ID,
  register(ctx: PluginContext) {
    ctx.registerProvider({
      id: PROVIDER_ID,
      credentialsForm: [
        { key: "accessToken", label: "Access token", kind: "secret" },
        { key: "refreshToken", label: "Refresh token", kind: "secret" },
        { key: "projectId", label: "Project id", kind: "string" },
        { key: "email", label: "Email", kind: "string" },
      ],
      normalizeCredentials,
      startOAuth,
      exchangeOAuth,
      keepalive,
      subscription,
      chatCompletions,
      responses,
      listModels,
      imageGenerations,
      imageEdits,
    });
  },
};

export default plugin;

export function normalizeCredentials(input: unknown): CredentialsWrite {
  const record = asRecord(input);
  if (!record) throw new Error("credentials must be an object");
  const accessToken = firstString(record.accessToken, record.access_token, record.token);
  if (!accessToken) throw new Error("accessToken is required");
  const refreshToken = firstString(record.refreshToken, record.refresh_token);
  const email = firstString(record.email);
  const projectId = firstString(record.projectId, nestedProjectId(record));
  const tier = firstString(record.tier);
  const expiresAt = expiresAtValue(record);
  const credentials: AntigravityCredentials = {
    accessToken,
    ...(refreshToken ? { refreshToken } : {}),
    ...(expiresAt ? { expiresAt } : {}),
    ...(email ? { email } : {}),
    ...(projectId ? { projectId } : {}),
    ...(tier ? { tier } : {}),
  };
  return { credentials, displayLabel: email ?? `Antigravity …${accessToken.slice(-6)}` };
}

export async function startOAuth(ctx: OAuthStartContext): Promise<OAuthStartResult> {
  const callback = new URL(ctx.callbackUrl);
  const state = callback.searchParams.get("state") || crypto.randomUUID();
  const redirectUri = await listenForLoopback(ctx.callbackUrl, state);
  const params = new URLSearchParams({
    client_id: ANTIGRAVITY_CLIENT_ID,
    response_type: "code",
    redirect_uri: redirectUri,
    scope: SCOPES.join(" "),
    state,
    access_type: "offline",
    prompt: "consent",
  });
  return {
    type: "authorization_url",
    authorizationUrl: `${AUTHORIZE_URL}?${params.toString()}`,
    session: { redirectUri },
  };
}

export async function exchangeOAuth(input: OAuthExchangeInput): Promise<CredentialsWrite> {
  const session = asRecord(input.session);
  const redirectUri = firstString(session?.redirectUri);
  const code = firstString(input.code);
  if (!redirectUri || !code) throw new Error("Antigravity OAuth callback is missing code or redirect");
  const token = await requestToken({
    grant_type: "authorization_code",
    client_id: ANTIGRAVITY_CLIENT_ID,
    client_secret: ANTIGRAVITY_CLIENT_SECRET,
    code,
    redirect_uri: redirectUri,
  });
  const accessToken = firstString(token.access_token);
  if (!accessToken) throw new Error("Antigravity token response did not include an access token");
  const discovered = await discoverAccount(accessToken).catch(() => ({
    email: undefined,
    projectId: undefined,
    tier: undefined,
  }));
  return normalizeCredentials({
    accessToken,
    refreshToken: firstString(token.refresh_token),
    expires_in: token.expires_in,
    email: discovered.email,
    projectId: discovered.projectId,
    tier: discovered.tier,
  });
}

export async function keepalive(input: KeepaliveInput): Promise<KeepaliveResult> {
  const credentials = readCredentials(input.credentials);
  const dueMs = millisecondsUntilRefresh(credentials.expiresAt);
  if (dueMs > 0) return { status: "ok", nextDueMs: dueMs };
  if (!credentials.refreshToken) {
    const error = "Antigravity access token expired";
    input.log.warn({ status: "needs_reauth", error }, "token refresh failed");
    return { status: "needs_reauth", error };
  }
  let token: Record<string, unknown>;
  try {
    token = await requestToken({
      grant_type: "refresh_token",
      client_id: ANTIGRAVITY_CLIENT_ID,
      client_secret: ANTIGRAVITY_CLIENT_SECRET,
      refresh_token: credentials.refreshToken,
    }, input.signal);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const status = / \(400\)| \(401\)/.test(message) ? "needs_reauth" : "failed";
    input.log.warn({ status, error: message }, "token refresh failed");
    return { status, error: message };
  }
  const normalized = normalizeCredentials({
    accessToken: firstString(token.access_token) ?? credentials.accessToken,
    refreshToken: firstString(token.refresh_token) ?? credentials.refreshToken,
    expires_in: token.expires_in,
    email: credentials.email,
    projectId: credentials.projectId,
    tier: credentials.tier,
  });
  const refreshed = readCredentials(normalized.credentials);
  return {
    status: "ok",
    credentials: normalized.credentials,
    displayLabel: normalized.displayLabel,
    nextDueMs: millisecondsUntilRefresh(refreshed.expiresAt) || 50 * 60 * 1000,
  };
}

export async function subscription(input: SubscriptionInput): Promise<SubscriptionResult> {
  let credentials: AntigravityCredentials;
  try {
    credentials = readCredentials(input.credentials);
  } catch (error) {
    return subscriptionFailure(input, "failed", error instanceof Error ? error.message : "Antigravity credentials are invalid");
  }
  if (accessTokenExpired(credentials.expiresAt)) {
    return credentials.refreshToken
      ? subscriptionFailure(input, "failed", "Antigravity access token expired; waiting for refresh")
      : subscriptionFailure(input, "needs_reauth", "Antigravity access token expired");
  }
  if (!credentials.projectId) {
    return subscriptionFailure(input, "failed", "Antigravity connection is missing projectId; reconnect OAuth");
  }
  const body = { project: credentials.projectId };
  const quota = await postJson(QUOTA_URLS, body, credentials.accessToken, input.signal);
  if (quota.status === 401) {
    return credentials.refreshToken
      ? subscriptionFailure(input, "failed", "Antigravity access token was rejected")
      : subscriptionFailure(input, "needs_reauth", "Antigravity session expired");
  }
  if (quota.status === 403) return subscriptionFailure(input, "failed", "Antigravity subscription unavailable");
  if (quota.status < 200 || quota.status >= 300) {
    return subscriptionFailure(input, "failed", `Antigravity quota request failed${quota.status ? ` (${quota.status})` : ""}`);
  }
  const [summary, catalog] = await Promise.all([
    postJson(QUOTA_SUMMARY_URLS, body, credentials.accessToken, input.signal),
    postCatalog(body, credentials.accessToken, input.signal),
  ]);
  return {
    status: "ok",
    plan: antigravityPlan(credentials.tier),
    usages: antigravityUsages(quota.payload, summary.status === 200 ? summary.payload : null, catalog.status === 200 ? catalog.payload : null),
    nextDueMs: SUBSCRIPTION_DUE_MS,
  };
}

export async function chatCompletions(
  request: JsonPostRequest,
  connection: ProviderConnection,
): Promise<HttpExchange> {
  const body = asRecord(request.body) ?? {};
  const credentials = readCredentials(connection.credentials);
  const projectId = credentials.projectId || (await discoverAccount(credentials.accessToken, request.signal)).projectId;
  if (!projectId) {
    throw Object.assign(new Error("Antigravity connection is missing projectId; reconnect OAuth"), { statusCode: 401 });
  }
  const stream = body.stream === true;
  const model = typeof body.model === "string" ? body.model : "";
  const envelope = buildAntigravityEnvelope(body, projectId, credentials.email || credentials.projectId || connection.connectionId);
  const upstream = await postFirstOk(stream, envelope, credentials.accessToken, request.signal);
  if (!upstream.ok || !stream) {
    if (!upstream.ok) {
      return { status: upstream.status, headers: pickHeaders(upstream.headers), body: upstream.body };
    }
    const payload = await upstream.json().catch(() => null) as unknown;
    const completion = toOpenAIChatCompletion(model, payload);
    return {
      status: 200,
      headers: { "content-type": "application/json" },
      body: new Response(JSON.stringify(completion)).body,
      parseUsage: parseOpenAIChatUsage,
      upstreamModel: envelope.model,
    };
  }
  return {
    status: 200,
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
    body: streamOpenAI(upstream.body, model),
    parseUsage: parseOpenAIChatUsage,
    upstreamModel: envelope.model,
  };
}

export async function responses(
  request: JsonPostRequest,
  connection: ProviderConnection,
): Promise<HttpExchange> {
  const converted = responsesRequestToChat(asRecord(request.body) ?? {});
  if (!converted.ok) return jsonError(converted.status, converted.message);
  const credentials = readCredentials(connection.credentials);
  const projectId = credentials.projectId || (await discoverAccount(credentials.accessToken, request.signal)).projectId;
  if (!projectId) {
    throw Object.assign(new Error("Antigravity connection is missing projectId; reconnect OAuth"), { statusCode: 401 });
  }
  const stream = converted.chat.stream === true;
  const model = typeof converted.chat.model === "string" ? converted.chat.model : "";
  const envelope = buildAntigravityEnvelope(
    converted.chat,
    projectId,
    credentials.email || credentials.projectId || connection.connectionId,
  );
  const upstream = await postFirstOk(stream, envelope, credentials.accessToken, request.signal);
  if (!upstream.ok || !stream) {
    if (!upstream.ok) {
      return { status: upstream.status, headers: pickHeaders(upstream.headers), body: upstream.body };
    }
    const payload = await upstream.json().catch(() => null) as unknown;
    const response = chatCompletionToResponse(model, toOpenAIChatCompletion(model, payload));
    return {
      status: 200,
      headers: { "content-type": "application/json" },
      body: new Response(JSON.stringify(response)).body,
      parseUsage: parseOpenAIResponsesUsage,
      upstreamModel: envelope.model,
    };
  }
  return {
    status: 200,
    headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
    body: streamResponses(upstream.body, model),
    parseUsage: parseOpenAIResponsesUsage,
    upstreamModel: envelope.model,
  };
}

export async function listModels(
  request: JsonGetRequest,
  connection: ProviderConnection,
): Promise<ModelList> {
  const credentials = readCredentials(connection.credentials);
  let lastStatus = 0;
  for (const url of MODEL_URLS) {
    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        headers: contentHeaders(credentials.accessToken),
        body: "{}",
        signal: request.signal,
      });
    } catch (error) {
      if (isAbortError(error)) throw error;
      continue;
    }
    lastStatus = response.status;
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) break;
      continue;
    }
    const payload = await response.json().catch(() => null) as unknown;
    const models = modelEntries(payload);
    const listed = collapseAntigravityModels(models);
    if (models.length === 0) continue;
    if (!listed.some((model) => model.id === ANTIGRAVITY_IMAGE_MODEL)) {
      listed.push({
        id: ANTIGRAVITY_IMAGE_MODEL,
        object: "model",
        owned_by: "antigravity",
        name: "Gemini 3.1 Flash Image",
      });
    }
    return { object: "list", data: listed };
  }
  throw Object.assign(new Error(`Antigravity models request failed (${lastStatus || "unreachable"})`), { statusCode: 502 });
}

const ANTIGRAVITY_IMAGE_MODEL = "gemini-3.1-flash-image";
const IMAGE_ASPECT_RATIOS: Record<string, string> = {
  "256x256": "1:1",
  "512x512": "1:1",
  "1024x1024": "1:1",
  "1792x1024": "16:9",
  "1024x1792": "9:16",
  "1536x1024": "3:2",
  "1024x1536": "2:3",
};

export async function imageGenerations(
  request: JsonPostRequest,
  connection: ProviderConnection,
): Promise<HttpExchange> {
  return generateImage(request, connection, false);
}

export async function imageEdits(
  request: JsonPostRequest,
  connection: ProviderConnection,
): Promise<HttpExchange> {
  return generateImage(request, connection, true);
}

async function generateImage(
  request: JsonPostRequest,
  connection: ProviderConnection,
  edit: boolean,
): Promise<HttpExchange> {
  const body = asRecord(request.body) ?? {};
  const requested = firstString(body.model);
  const prompt = firstString(body.prompt) ?? "";
  const references = referenceImages(body);
  if (!requested || !prompt) return imageError(400, "model and prompt are required");
  if (edit && references.length === 0) return imageError(400, "image is required");
  const credentials = readCredentials(connection.credentials);
  const projectId = credentials.projectId || (await discoverAccount(credentials.accessToken, request.signal)).projectId;
  if (!projectId) {
    throw Object.assign(new Error("Antigravity connection is missing projectId; reconnect OAuth"), { statusCode: 401 });
  }
  const slash = requested.lastIndexOf("/");
  const model = (slash >= 0 ? requested.slice(slash + 1) : requested).trim();
  const count = typeof body.n === "number" && Number.isFinite(body.n) && body.n > 0 ? Math.floor(body.n) : 1;
  const envelope = {
    project: projectId,
    requestId: `image_gen/${Date.now()}/${crypto.randomUUID()}/0`,
    request: {
      contents: [{
        role: "user",
        parts: [
          ...references.map((image) => ({ inlineData: { mimeType: image.mediaType, data: image.data } })),
          { text: prompt },
        ],
      }],
      generationConfig: {
        candidateCount: count,
        imageConfig: { aspectRatio: aspectRatio(body) },
      },
    },
    model,
    userAgent: "antigravity",
    requestType: "image_gen",
  };
  const upstream = await postFirstOk(false, envelope, credentials.accessToken, request.signal);
  if (!upstream.ok) {
    return { status: upstream.status, headers: pickHeaders(upstream.headers), body: upstream.body };
  }
  const images = generatedImages(await upstream.json().catch(() => null), prompt);
  if (images.length === 0) return imageError(502, "Antigravity image generation returned no images");
  return {
    status: 200,
    headers: { "content-type": "application/json" },
    body: new Response(JSON.stringify({ created: Math.floor(Date.now() / 1000), data: images })).body,
    upstreamModel: model,
  };
}

function aspectRatio(body: Record<string, unknown>): string {
  const explicit = firstString(body.aspect_ratio);
  if (explicit && /^\d+:\d+$/.test(explicit)) return explicit;
  const size = firstString(body.size);
  if (!size) return "1:1";
  if (/^\d+:\d+$/.test(size)) return size;
  return IMAGE_ASPECT_RATIOS[size] ?? "1:1";
}

function referenceImages(body: Record<string, unknown>): Array<{ mediaType: string; data: string }> {
  if (!Array.isArray(body.images)) return [];
  const images: Array<{ mediaType: string; data: string }> = [];
  for (const entry of body.images) {
    const image = asRecord(entry);
    const data = firstString(image?.b64_json);
    if (!data) continue;
    images.push({ mediaType: firstString(image?.media_type) ?? "image/png", data });
  }
  return images;
}

function generatedImages(payload: unknown, prompt: string): Array<{ b64_json: string; revised_prompt: string }> {
  const records = Array.isArray(payload) ? payload : [payload];
  const images: Array<{ b64_json: string; revised_prompt: string }> = [];
  for (const record of records) {
    const wrapped = asRecord(record);
    const response = asRecord(wrapped?.response) ?? wrapped;
    const candidates = Array.isArray(response?.candidates) ? response.candidates : [];
    for (const candidate of candidates) {
      const content = asRecord(asRecord(candidate)?.content);
      const parts = content && Array.isArray(content.parts) ? content.parts : [];
      let revised = prompt;
      for (const part of parts) {
        const text = firstString(asRecord(part)?.text);
        if (text) revised = text;
      }
      for (const part of parts) {
        const data = firstString(asRecord(asRecord(part)?.inlineData)?.data);
        if (data) images.push({ b64_json: data, revised_prompt: revised });
      }
    }
  }
  return images;
}

function jsonError(status: number, message: string): HttpExchange {
  return {
    status,
    headers: { "content-type": "application/json" },
    body: new Response(JSON.stringify({ error: { message, type: "invalid_request_error" } })).body,
  };
}

function imageError(status: number, message: string): HttpExchange {
  return jsonError(status, message);
}

export async function closeAntigravityLoopback(): Promise<void> {
  const current = loopback;
  loopback = undefined;
  if (!current) return;
  clearTimeout(current.timeout);
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  current.server.close((error) => {
    if (error) reject(error);
    else resolve();
  });
  await promise;
}

type UpstreamJson = { status: number; payload: unknown };

async function postJson(
  urls: string[],
  body: unknown,
  accessToken: string,
  signal: AbortSignal,
): Promise<UpstreamJson> {
  let lastStatus = 0;
  for (const url of urls) {
    let response: Response;
    try {
      response = await fetch(url, {
        method: "POST",
        redirect: "manual",
        headers: contentHeaders(accessToken),
        body: JSON.stringify(body),
        signal,
      });
    } catch (error) {
      if (isAbortError(error)) throw error;
      continue;
    }
    lastStatus = response.status;
    if (response.status === 401 || response.status === 403) return { status: response.status, payload: null };
    if (!response.ok) continue;
    return { status: response.status, payload: await response.json().catch(() => null) };
  }
  return { status: lastStatus, payload: null };
}

async function postCatalog(body: unknown, accessToken: string, signal: AbortSignal): Promise<UpstreamJson> {
  let lastStatus = 0;
  let lastPayload: unknown = null;
  for (const url of MODEL_URLS) {
    const result = await postJson([url], body, accessToken, signal);
    lastStatus = result.status;
    if (result.status === 401 || result.status === 403) return result;
    if (result.status < 200 || result.status >= 300) continue;
    lastPayload = result.payload;
    if (hasModels(result.payload)) return result;
  }
  return { status: lastStatus, payload: lastPayload };
}

function hasModels(payload: unknown): boolean {
  const models = asRecord(payload)?.models;
  if (Array.isArray(models)) return models.length > 0;
  if (!models || typeof models !== "object") return false;
  return Object.keys(models).length > 0;
}

function subscriptionFailure(
  input: SubscriptionInput,
  status: "failed" | "needs_reauth",
  error: string,
): SubscriptionResult {
  input.log.warn({ status, error }, "subscription refresh failed");
  return { status, error };
}

function accessTokenExpired(expiresAt: string | undefined): boolean {
  if (!expiresAt) return false;
  const expires = Date.parse(expiresAt);
  return !Number.isNaN(expires) && expires <= Date.now();
}

async function postFirstOk(
  stream: boolean,
  envelope: unknown,
  accessToken: string,
  signal: AbortSignal | undefined,
): Promise<Response> {
  let lastError: unknown;
  for (const base of RUNTIME_BASES) {
    try {
      const response = await fetch(antigravityUrl(base, stream), {
        method: "POST",
        headers: contentHeaders(accessToken),
        body: JSON.stringify(envelope),
        signal,
      });
      if (response.ok || response.status < 500) return response;
      lastError = new Error(`Antigravity upstream failed (${response.status})`);
    } catch (error) {
      if (isAbortError(error)) throw error;
      lastError = error;
    }
  }
  throw new Error(`upstream request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

function streamOpenAI(body: ReadableStream<Uint8Array> | null, model: string): ReadableStream<Uint8Array> | null {
  if (!body) return null;
  const state = createStreamState();
  let pending = "";
  const decoder = new TextDecoder();
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      pending += decoder.decode(chunk, { stream: true });
      const taken = takeSseEvents(pending);
      pending = taken.rest;
      const frames = taken.events.flatMap((event) => toOpenAIStreamChunks(model, event, state));
      if (frames.length > 0) controller.enqueue(encodeSse(frames));
    },
    flush(controller) {
      pending += decoder.decode();
      if (pending.trim()) {
        const taken = takeSseEvents(`${pending}\n\n`);
        const frames = taken.events.flatMap((event) => toOpenAIStreamChunks(model, event, state));
        if (frames.length > 0) controller.enqueue(encodeSse(frames));
      }
      controller.enqueue(encodeSse([], true));
    },
  }));
}

async function discoverAccount(
  accessToken: string,
  signal?: AbortSignal,
): Promise<{ email?: string; projectId?: string; tier?: string }> {
  const headers = nodeHeaders(accessToken);
  const [userInfo, loaded] = await Promise.all([
    fetch(USERINFO_URL, { headers: { authorization: `Bearer ${accessToken}` }, signal }).catch(() => null),
    fetch(LOAD_CODE_ASSIST_URLS[0], {
      method: "POST",
      headers,
      body: JSON.stringify({ metadata: { ideType: "ANTIGRAVITY" } }),
      signal,
    }).catch(() => null),
  ]);
  const profile = userInfo?.ok ? asRecord(await userInfo.json().catch(() => null)) : undefined;
  const data = loaded?.ok ? asRecord(await loaded.json().catch(() => null)) : undefined;
  const paid = asRecord(data?.paidTier);
  const current = asRecord(data?.currentTier);
  return {
    email: firstString(profile?.email),
    projectId: projectIdFrom(data),
    tier: firstString(paid?.id, current?.id),
  };
}

async function requestToken(body: Record<string, string>, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const response = await fetch(TOKEN_URL, {
    method: "POST",
    headers: {
      "content-type": "application/x-www-form-urlencoded",
      accept: "application/json",
      "user-agent": nodeUserAgent(),
    },
    body: new URLSearchParams(body),
    signal,
  });
  const payload = asRecord(await response.json().catch(() => null)) ?? {};
  if (!response.ok) {
    const detail = firstString(payload.error_description, payload.error);
    throw new Error(`Antigravity token request failed (${response.status})${detail ? `: ${detail}` : ""}`);
  }
  return payload;
}

async function listenForLoopback(callbackUrl: string, state: string): Promise<string> {
  await closeAntigravityLoopback();
  const server = createServer((req, res) => {
    const requestUrl = new URL(req.url ?? "/", `http://${LOOPBACK_HOST}`);
    if (requestUrl.pathname !== LOOPBACK_PATH) {
      res.writeHead(404).end();
      return;
    }
    const next = new URL(callbackUrl);
    const code = requestUrl.searchParams.get("code");
    const nextState = requestUrl.searchParams.get("state") ?? state;
    if (code) next.searchParams.set("code", code);
    if (nextState) next.searchParams.set("state", nextState);
    res.writeHead(302, { location: next.toString() }).end();
    void closeAntigravityLoopback();
  });
  const { promise, resolve, reject } = Promise.withResolvers<number>();
  server.once("error", reject);
  server.listen(0, LOOPBACK_HOST, () => {
    const address = server.address();
    if (!address || typeof address === "string") {
      reject(new Error("Antigravity OAuth loopback did not bind"));
      return;
    }
    resolve(address.port);
  });
  const port = await promise;
  const timeout = setTimeout(() => {
    void closeAntigravityLoopback();
  }, LOOPBACK_TIMEOUT_MS);
  timeout.unref();
  loopback = { server, timeout };
  return `http://${LOOPBACK_HOST}:${port}${LOOPBACK_PATH}`;
}

function contentHeaders(accessToken: string): Record<string, string> {
  return {
    authorization: `Bearer ${accessToken}`,
    "content-type": "application/json",
    "user-agent": cloudCodeUserAgent(),
  };
}

function nodeHeaders(accessToken: string): Record<string, string> {
  return {
    authorization: `Bearer ${accessToken}`,
    "content-type": "application/json",
    "user-agent": cloudCodeUserAgent(),
    "x-goog-api-client": "gl-node/22.21.1",
  };
}

function cloudCodeUserAgent(): string {
  return `antigravity/hub/${HUB_VERSION} darwin/arm64`;
}

function nodeUserAgent(): string {
  return `antigravity/hub/${HUB_VERSION} darwin/arm64 google-api-nodejs-client/10.3.0`;
}

function modelEntries(payload: unknown): ModelList["data"] {
  const record = asRecord(payload);
  const models = record?.models;
  const entries: ModelList["data"] = [];
  const seen: Record<string, true> = {};
  const push = (id: string, name?: string) => {
    const trimmed = id.trim();
    if (!trimmed || seen[trimmed]) return;
    seen[trimmed] = true;
    entries.push({ id: trimmed, object: "model", owned_by: "antigravity", ...(name && name !== trimmed ? { name } : {}) });
  };
  if (Array.isArray(models)) {
    for (const entry of models) {
      const item = asRecord(entry);
      const id = firstString(item?.id, item?.name);
      if (!id) continue;
      push(id, firstString(item?.displayName, item?.name));
    }
  } else if (models && typeof models === "object") {
    for (const [id, info] of Object.entries(models)) {
      const item = asRecord(info);
      if (item?.isInternal === true) continue;
      push(id, firstString(item?.displayName, item?.name));
    }
  }
  return entries;
}

function readCredentials(input: unknown): AntigravityCredentials {
  return normalizeCredentials(input).credentials as AntigravityCredentials;
}

function projectIdFrom(data: Record<string, unknown> | undefined): string | undefined {
  const project = data?.cloudaicompanionProject;
  if (typeof project === "string") return project.trim() || undefined;
  return firstString(asRecord(project)?.id);
}

function nestedProjectId(record: Record<string, unknown>): unknown {
  return asRecord(record.providerSpecificData)?.projectId;
}

function expiresAtValue(record: Record<string, unknown>): string | undefined {
  const explicit = firstString(record.expiresAt, record.expires_at);
  if (explicit) return explicit;
  const seconds = typeof record.expires_in === "number" ? record.expires_in : typeof record.expiresIn === "number" ? record.expiresIn : undefined;
  if (!seconds || !Number.isFinite(seconds)) return undefined;
  return new Date(Date.now() + seconds * 1000).toISOString();
}

function millisecondsUntilRefresh(expiresAt: string | undefined): number {
  if (!expiresAt) return 5 * 60 * 1000;
  const expires = Date.parse(expiresAt);
  if (Number.isNaN(expires)) return 0;
  return Math.max(0, expires - Date.now() - REFRESH_SKEW_MS);
}

function pickHeaders(headers: Headers): Record<string, string> {
  const picked: Record<string, string> = {};
  for (const name of ["content-type", "cache-control"]) {
    const value = headers.get(name);
    if (value) picked[name] = value;
  }
  return picked;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}

function firstString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
