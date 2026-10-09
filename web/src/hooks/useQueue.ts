import { useCallback, useEffect, useRef, useState } from "react";
import { createAnalysis, getAnalysis } from "../api/client";
import type { Verdict } from "../types/api";

export type QueueState = "waiting" | "uploading" | "analysing" | "done" | "failed";

export interface QueueItem {
  key: string;
  name: string;
  state: QueueState;
  jobId?: string;
  verdict?: Verdict;
  probability?: number;
  error?: string;
}

/**
 * Several clips at once.
 *
 * Uploads go one at a time (parallel 100 MB uploads would only compete for the
 * same connection); the server's worker pool does the analysis, and one poll
 * loop tracks every unfinished job.
 */
export function useQueue() {
  const [items, setItems] = useState<QueueItem[]>([]);
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const poll = useRef<number | null>(null);

  const patch = useCallback((key: string, p: Partial<QueueItem>) => {
    setItems((xs) => xs.map((x) => (x.key === key ? { ...x, ...p } : x)));
  }, []);

  const stopPolling = useCallback(() => {
    if (poll.current) window.clearInterval(poll.current);
    poll.current = null;
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const tick = useCallback(async () => {
    const pending = itemsRef.current.filter((x) => x.state === "analysing" && x.jobId);
    if (!pending.length && !itemsRef.current.some((x) => x.state === "waiting" || x.state === "uploading")) {
      stopPolling();
      return;
    }
    await Promise.all(
      pending.map(async (x) => {
        try {
          const s = await getAnalysis(x.jobId!);
          if (s.state === "done" && s.result) {
            patch(x.key, {
              state: "done",
              verdict: s.result.verdict,
              probability: s.result.calibrated_probability,
            });
          } else if (s.state === "failed" || s.state === "cancelled") {
            patch(x.key, { state: "failed", error: s.error ?? "Analysis failed." });
          }
        } catch {
          /* transient; next tick */
        }
      }),
    );
  }, [patch, stopPolling]);

  const enqueue = useCallback(
    async (files: File[]) => {
      const added: QueueItem[] = files.map((f, i) => ({
        key: `${Date.now()}-${i}-${f.name}`,
        name: f.name,
        state: "waiting",
      }));
      setItems((xs) => [...xs, ...added]);
      if (!poll.current) poll.current = window.setInterval(() => void tick(), 2000);

      for (let i = 0; i < files.length; i++) {
        const key = added[i].key;
        patch(key, { state: "uploading" });
        try {
          const { job_id } = await createAnalysis(files[i]);
          patch(key, { state: "analysing", jobId: job_id });
        } catch (e) {
          patch(key, { state: "failed", error: e instanceof Error ? e.message : String(e) });
        }
      }
    },
    [patch, tick],
  );

  const clear = useCallback(() => {
    stopPolling();
    setItems([]);
  }, [stopPolling]);

  return { items, enqueue, clear };
}
