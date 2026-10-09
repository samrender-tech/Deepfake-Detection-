import { useCallback, useEffect, useRef, useState } from "react";
import {
  createAnalysis,
  getAnalysis,
  subscribeEvents,
} from "../api/client";
import type { AnalysisStatus, Stage } from "../types/api";

type Phase = "idle" | "uploading" | "analysing" | "done" | "error";

export function useAnalysis() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [uploadFraction, setUploadFraction] = useState(0);
  const [stage, setStage] = useState<Stage>("queued");
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const cleanup = useRef<(() => void) | null>(null);
  const poll = useRef<number | null>(null);

  const stop = useCallback(() => {
    cleanup.current?.();
    cleanup.current = null;
    if (poll.current) window.clearInterval(poll.current);
    poll.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const reset = useCallback(() => {
    stop();
    setPhase("idle");
    setUploadFraction(0);
    setStage("queued");
    setProgress(0);
    setStatus(null);
    setError(null);
    setJobId(null);
  }, [stop]);

  const finish = useCallback(
    async (jobId: string) => {
      stop();
      try {
        const s = await getAnalysis(jobId);
        setStatus(s);
        if (s.state === "failed") {
          setPhase("error");
          setError(s.error ?? "Analysis failed.");
        } else {
          setPhase("done");
        }
      } catch (e) {
        setPhase("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [stop],
  );

  const analyse = useCallback(
    async (file: File) => {
      reset();
      setPhase("uploading");
      try {
        const { job_id } = await createAnalysis(file, setUploadFraction);
        setJobId(job_id);
        setPhase("analysing");

        cleanup.current = subscribeEvents(job_id, (ev) => {
          setStage(ev.stage as Stage);
          setProgress(ev.progress);
          if (ev.state === "done" || ev.state === "failed") void finish(job_id);
        });

        // Polling fallback: SSE can be dropped by a proxy, and the user must
        // still get their result. 2s is frequent enough to feel immediate and
        // cheap enough to be harmless.
        poll.current = window.setInterval(async () => {
          try {
            const s = await getAnalysis(job_id);
            setStage(s.stage);
            setProgress(s.progress);
            if (s.state === "done" || s.state === "failed") void finish(job_id);
          } catch {
            /* transient; SSE or the next tick will catch up */
          }
        }, 2000);
      } catch (e) {
        setPhase("error");
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [reset, finish],
  );

  /** Re-open a finished analysis from the history list. */
  const open = useCallback(
    async (id: string) => {
      reset();
      setJobId(id);
      setPhase("analysing");
      await finish(id);
    },
    [reset, finish],
  );

  return {
    phase,
    uploadFraction,
    stage,
    progress,
    status,
    error,
    jobId,
    analyse,
    open,
    reset,
  };
}
