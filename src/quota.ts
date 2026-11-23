import type { SubscriptionUsage } from "@viateai/sdk";

type ModelQuota = {
  id: string;
  label: string;
  fraction: number | undefined;
  resetAt: string | undefined;
  internal: boolean;
};

export function antigravityPlan(tier: string | undefined): string {
  const raw = tier?.trim();
  if (!raw) return "Antigravity";
  const upper = raw.toUpperCase();
  if (upper.includes("ULTRA")) return "Ultra";
  if (upper.includes("ENTERPRISE")) return "Enterprise";
  if (upper.includes("PRO") || upper.includes("PREMIUM") || upper.includes("GOOGLE_ONE") || upper.includes("ONE_AI")) return "Pro";
  if (upper.includes("BUSINESS") || upper.includes("STANDARD")) return "Business";
  if (upper.includes("PLUS")) return "Plus";
  if (upper.includes("LITE") || upper.includes("LIGHT")) return "Lite";
  if (upper.includes("FREE") || upper.includes("INDIVIDUAL") || upper.includes("LEGACY")) return "Free";
  return raw.slice(0, 120);
}

export function antigravityUsages(quota: unknown, summary: unknown, catalog: unknown): SubscriptionUsage[] {
  const catalogModels = readCatalog(catalog);
  const known = new Map(catalogModels.map((model) => [model.id, model]));
  const usages: SubscriptionUsage[] = [];
  const seen = new Set<string>();
  for (const bucket of readLive(quota)) {
    const match = known.get(bucket.id);
    if (match?.internal || bucket.fraction === undefined || seen.has(bucket.id)) continue;
    seen.add(bucket.id);
    const usage = toPercent(bucket.id, match?.label || bucket.id, "Models", bucket.fraction, bucket.resetAt);
    if (usage) usages.push(usage);
  }
  for (const model of catalogModels) {
    if (seen.has(model.id) || model.internal || model.fraction === undefined) continue;
    // fetchAvailableModels keeps reporting a full window after real use. A live bucket is the consumption signal.
    if (model.fraction >= 1 && model.resetAt) continue;
    seen.add(model.id);
    const usage = toPercent(model.id, model.label, "Models", model.fraction, model.resetAt);
    if (usage) usages.push(usage);
  }
  for (const weekly of readWeekly(summary)) {
    if (seen.has(weekly.type)) continue;
    seen.add(weekly.type);
    usages.push(weekly);
  }
  return usages;
}

function readLive(payload: unknown): Array<{ id: string; fraction: number | undefined; resetAt: string | undefined }> {
  const buckets = asRecord(payload)?.buckets;
  if (!Array.isArray(buckets)) return [];
  const models: Array<{ id: string; fraction: number | undefined; resetAt: string | undefined }> = [];
  for (const entry of buckets) {
    const bucket = asRecord(entry);
    const id = modelId(firstString(bucket?.modelId));
    if (!id) continue;
    models.push({ id, fraction: fraction(bucket?.remainingFraction), resetAt: resetTime(bucket?.resetTime) });
  }
  return models;
}

function readCatalog(payload: unknown): ModelQuota[] {
  const models = asRecord(payload)?.models;
  const entries: ModelQuota[] = [];
  if (Array.isArray(models)) {
    for (const entry of models) {
      const item = asRecord(entry);
      const id = modelId(firstString(item?.id, item?.name));
      if (!id) continue;
      entries.push(catalogModel(id, item));
    }
    return entries;
  }
  if (!models || typeof models !== "object") return entries;
  for (const [key, info] of Object.entries(models)) {
    const id = modelId(key);
    if (!id) continue;
    entries.push(catalogModel(id, asRecord(info)));
  }
  return entries;
}

function catalogModel(id: string, item: Record<string, unknown> | undefined): ModelQuota {
  const quota = asRecord(item?.quotaInfo);
  return {
    id,
    label: firstString(item?.displayName, item?.name) ?? id,
    fraction: fraction(quota?.remainingFraction),
    resetAt: resetTime(quota?.resetTime),
    internal: item?.isInternal === true,
  };
}

function readWeekly(payload: unknown): SubscriptionUsage[] {
  const root = asRecord(payload);
  const nested = asRecord(root?.quotaSummary);
  const groups = arrayOf(root?.groups) ?? arrayOf(nested?.groups);
  if (!groups) return [];
  const usages: SubscriptionUsage[] = [];
  for (const entry of groups) {
    const group = asRecord(entry);
    const label = firstString(group?.displayName);
    const buckets = arrayOf(group?.buckets);
    if (!label || !buckets) continue;
    const weekly = buckets.map(asRecord).find((bucket) => bucket && bucket.disabled !== true && isWeekly(bucket));
    const value = fraction(weekly?.remainingFraction);
    const type = weeklyType(label);
    if (!weekly || value === undefined || !type) continue;
    const usage = toPercent(type, label, "Weekly", value, resetTime(weekly.resetTime));
    if (usage) usages.push(usage);
  }
  return usages;
}

function toPercent(
  type: string,
  label: string,
  group: string,
  value: number,
  resetAt: string | undefined,
): SubscriptionUsage | undefined {
  if (!type || type.length > 200) return undefined;
  const safeLabel = (label.trim() || type).slice(0, 200);
  if (!safeLabel) return undefined;
  if (value >= 1 && !resetAt) {
    return { type, label: safeLabel, group, unit: "percent", used: 0, limit: null, remaining: null, unlimited: true, reported: true };
  }
  const remaining = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return {
    type,
    label: safeLabel,
    group,
    unit: "percent",
    used: 100 - remaining,
    limit: 100,
    remaining,
    ...(resetAt ? { resetsAt: resetAt } : {}),
    reported: true,
  };
}

function isWeekly(bucket: Record<string, unknown>): boolean {
  return /\bweekly\b/.test(`${firstString(bucket.bucketId) ?? ""} ${firstString(bucket.displayName) ?? ""}`.toLowerCase());
}

function weeklyType(displayName: string): string | undefined {
  const cleaned = displayName
    .toLowerCase()
    .replace(/\bmodels?\b/g, "")
    .replace(/\band\b/g, " ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!cleaned) return undefined;
  return `weekly:${cleaned}`.slice(0, 200);
}

function modelId(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const slash = value.lastIndexOf("/");
  const id = (slash >= 0 ? value.slice(slash + 1) : value).trim();
  return id || undefined;
}

function fraction(value: unknown): number | undefined {
  const parsed = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : Number.NaN;
  if (!Number.isFinite(parsed)) return undefined;
  return Math.min(1, Math.max(0, parsed));
}

function resetTime(value: unknown): string | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  let parsed: number;
  if (typeof value === "number") {
    parsed = value < 1e12 ? value * 1000 : value;
  } else if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    const numeric = Number(value.trim());
    parsed = numeric < 1e12 ? numeric * 1000 : numeric;
  } else if (typeof value === "string") {
    parsed = Date.parse(value);
  } else {
    return undefined;
  }
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return new Date(parsed).toISOString();
}

function arrayOf(value: unknown): unknown[] | undefined {
  return Array.isArray(value) ? value : undefined;
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
