import { useEffect, useState } from "react";
import { getLimits } from "./api/client";
import { Badges } from "./components/Badges";
import { Evidence } from "./components/Evidence";
import { Explanation } from "./components/Explanation";
import { History } from "./components/History";
import { FrameTimeline } from "./components/FrameTimeline";
import { ModelPanel } from "./components/ModelPanel";
import { Queue } from "./components/Queue";
import { Progress } from "./components/Progress";
import { ReportActions } from "./components/ReportActions";
import { StreamBars } from "./components/StreamBars";
import { SyncChart } from "./components/SyncChart";
import { Upload } from "./components/Upload";
import { VerdictCard } from "./components/VerdictCard";
import { useAnalysis } from "./hooks/useAnalysis";
import { useQueue } from "./hooks/useQueue";
import type { Limits } from "./types/api";

export default function App() {
  const [limits, setLimits] = useState<Limits | null>(null);
  const [limitsError, setLimitsError] = useState<string | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<number | null>(null);
  const [filename, setFilename] = useState<string>();
  const [historyKey, setHistoryKey] = useState(0);

  const { phase, uploadFraction, stage, progress, status, error, jobId, analyse, open, reset } =
    useAnalysis();
  const queue = useQueue();

  useEffect(() => {
    getLimits()
      .then(setLimits)
      .catch((e) => setLimitsError(e instanceof Error ? e.message : String(e)));
  }, []);

  const result = status?.result ?? null;
  const busy = phase === "uploading" || phase === "analysing";

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
      <header>
        <p className="text-[11px] uppercase tracking-[0.2em] text-violet-300/80">
          AVFORGE · research prototype
        </p>
        <h1 className="mt-1.5 text-2xl font-semibold tracking-tight sm:text-3xl">
          Audio-visual deepfake detection
        </h1>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-neutral-400">
          Upload a short clip. The detector reports a calibrated probability, and
          says so when it is not confident enough to call it.
        </p>
      </header>

      {limitsError && (
        <p role="alert" className="mt-6 rounded-lg bg-amber-500/10 p-3 text-xs text-amber-200">
          Could not reach the API ({limitsError}). Start it with{" "}
          <code className="rounded bg-black/40 px-1">make serve</code>.
        </p>
      )}

      <main className="mt-8 space-y-4">
        {phase === "idle" && (
          <>
            <Upload
              limits={limits}
              onFile={(f) => {
                setFilename(f.name);
                setSelectedFrame(null);
                void analyse(f);
              }}
              onFiles={(fs) => void queue.enqueue(fs)}
            />
            <Queue
              items={queue.items}
              onOpen={(id) => {
                setFilename(undefined);
                setSelectedFrame(null);
                void open(id);
              }}
              onClear={() => {
                queue.clear();
                setHistoryKey((k) => k + 1);
              }}
            />
            <History
              refreshKey={historyKey}
              onOpen={(id) => {
                setFilename(undefined);
                setSelectedFrame(null);
                void open(id);
              }}
            />
            <ModelPanel />
          </>
        )}

        {busy && (
          <Progress
            stage={stage}
            progress={progress}
            uploadFraction={uploadFraction}
            uploading={phase === "uploading"}
            filename={filename}
          />
        )}

        {phase === "error" && (
          <section role="alert" className="rounded-xl bg-manipulated-bg p-4 ring-1 ring-manipulated-ring">
            <h2 className="text-sm font-medium text-manipulated-fg">Analysis failed</h2>
            <p className="mt-1.5 text-sm text-neutral-300">{error}</p>
            <button
              onClick={reset}
              className="mt-3 rounded-md bg-neutral-100 px-3 py-1.5 text-xs font-medium text-neutral-900 hover:bg-white"
            >
              Try another clip
            </button>
          </section>
        )}

        {phase === "done" && result && (
          <>
            {status?.filename && (
              <p className="truncate text-xs text-neutral-500">
                Results for <span className="text-neutral-300">{status.filename}</span>
              </p>
            )}
            <VerdictCard result={result} />
            <Badges result={result} />

            {result.warnings.length > 0 && (
              <section className="rounded-xl bg-amber-500/10 p-4 ring-1 ring-amber-500/25">
                <h3 className="text-sm font-medium text-amber-200">Caveats for this clip</h3>
                <ul className="mt-2 space-y-1.5 text-xs leading-relaxed text-amber-100/90">
                  {result.warnings.map((w, i) => (
                    <li key={i}>• {w}</li>
                  ))}
                </ul>
              </section>
            )}

            <Explanation result={result} />
            <StreamBars result={result} />
            <FrameTimeline
              result={result}
              selected={selectedFrame}
              onSelectFrame={setSelectedFrame}
            />
            <SyncChart result={result} />
            <Evidence result={result} selectedFrame={selectedFrame} />
            {jobId && <ReportActions jobId={jobId} />}

            <section className="rounded-xl bg-neutral-900/60 p-4 ring-1 ring-neutral-800">
              <h3 className="text-sm font-medium text-neutral-300">Limitations</h3>
              <p className="mt-2 text-xs leading-relaxed text-neutral-400">{result.disclaimer}</p>
              <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-neutral-500">
                <dt>Model</dt>
                <dd className="truncate font-mono">{result.model_version || "—"}</dd>
                <dt>Total time</dt>
                <dd className="tabular-nums">
                  {result.latency_ms.total_ms
                    ? `${(result.latency_ms.total_ms / 1000).toFixed(1)}s`
                    : "—"}
                </dd>
              </dl>
            </section>

            <div className="flex gap-2 pt-1">
              <button
                onClick={() => {
                  reset();
                  setFilename(undefined);
                  setSelectedFrame(null);
                  setHistoryKey((k) => k + 1);
                }}
                className="rounded-md bg-neutral-100 px-3.5 py-2 text-xs font-medium text-neutral-900 hover:bg-white"
              >
                Analyse another clip
              </button>
            </div>
          </>
        )}
      </main>

      <footer className="mt-14 border-t border-neutral-800 pt-5 text-[11px] leading-relaxed text-neutral-600">
        <p>
          A detector, not a generator: this project creates no synthetic media. Results
          are probabilistic and must not be used to make decisions about a person.
        </p>
      </footer>
    </div>
  );
}
