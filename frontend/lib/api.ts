import type {
  BenchmarkResult,
  PredictPayload,
  PredictionRequest,
  Prediction,
  RecentTriage,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    return JSON.stringify(body);
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}

function apiUnavailableError(): Error {
  return new Error(
    `Cannot reach the AegisKEV API at ${API_BASE}. Start it from the repository root with: uvicorn backend.app.main:app --reload --port 8000`,
  );
}

export async function predict(payload: PredictionRequest): Promise<Prediction> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
  } catch {
    throw apiUnavailableError();
  }
  if (!response.ok) {
    throw new Error(await errorDetail(response));
  }
  return (await response.json()) as Prediction;
}

export async function fetchBenchmark(): Promise<BenchmarkResult> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/metrics/benchmark`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return { ok: false, detail: await errorDetail(response) };
    }
    return { ok: true, data: await response.json() };
  } catch {
    return {
      ok: false,
      detail: apiUnavailableError().message,
    };
  }
}

export async function fetchRecentTriage(limit = 25): Promise<RecentTriage> {
  let response: Response;
  try {
    response = await fetch(
      `${API_BASE}/api/v1/triage/recent?limit=${encodeURIComponent(limit)}`,
      { cache: "no-store" },
    );
  } catch {
    throw apiUnavailableError();
  }
  if (!response.ok) {
    throw new Error(await errorDetail(response));
  }
  return (await response.json()) as RecentTriage;
}
