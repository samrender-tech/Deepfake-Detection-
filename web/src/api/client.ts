import type { AnalysisStatus, Limits, ModelInfo, ReportFormat } from "../types/api";

const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

function headers(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function unwrap<T>(r: Response): Promise<T> {
  if (!r.ok) {
    // The API returns {detail} for client errors; surface it verbatim because
    // these messages are written to be actionable ("clip is 90s, limit is 60s").
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export async function getLimits(): Promise<Limits> {
  return unwrap<Limits>(await fetch("/v1/limits", { headers: headers() }));
}

export async function createAnalysis(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<{ job_id: string }> {
  // XHR rather than fetch: fetch still has no upload-progress event, and a
  // 100 MB upload with no feedback feels broken.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/v1/analyses");
    if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let detail = `${xhr.status} ${xhr.statusText}`;
        try {
          detail = JSON.parse(xhr.responseText).detail ?? detail;
        } catch {
          /* ignore */
        }
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error("Network error during upload."));
    xhr.send(form);
  });
}

export async function getAnalysis(jobId: string): Promise<AnalysisStatus> {
  return unwrap<AnalysisStatus>(
    await fetch(`/v1/analyses/${jobId}`, { headers: headers() }),
  );
}

export async function listAnalyses(limit = 20): Promise<AnalysisStatus[]> {
  return unwrap<AnalysisStatus[]>(
    await fetch(`/v1/analyses?limit=${limit}`, { headers: headers() }),
  );
}

export async function getModelInfo(): Promise<ModelInfo> {
  return unwrap<ModelInfo>(await fetch("/v1/model", { headers: headers() }));
}

/**
 * Download a finished analysis as a file.
 *
 * Fetched rather than linked so the API key header is sent; the browser then
 * saves the blob under the name the server put in Content-Disposition.
 */
export async function downloadReport(jobId: string, format: ReportFormat): Promise<void> {
  const r = await fetch(`/v1/analyses/${jobId}/report?format=${format}`, {
    headers: headers(),
  });
  if (!r.ok) {
    await unwrap(r);
    return;
  }
  const match = /filename="([^"]+)"/.exec(r.headers.get("content-disposition") ?? "");
  const name = match?.[1] ?? `avforge-report.${format}`;
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function deleteAnalysis(jobId: string): Promise<void> {
  const r = await fetch(`/v1/analyses/${jobId}`, {
    method: "DELETE",
    headers: headers(),
  });
  if (!r.ok && r.status !== 404) throw new Error(`Delete failed: ${r.status}`);
}

/** Subscribe to stage progress. Returns an unsubscribe function. */
export function subscribeEvents(
  jobId: string,
  onEvent: (e: { state: string; stage: string; progress: number }) => void,
): () => void {
  const es = new EventSource(`/v1/analyses/${jobId}/events`);
  es.onmessage = (ev) => {
    try {
      onEvent(JSON.parse(ev.data));
    } catch {
      /* keepalive comment lines */
    }
  };
  // EventSource retries on its own; the polling fallback in useAnalysis is
  // what actually guarantees the result arrives.
  es.onerror = () => es.close();
  return () => es.close();
}
