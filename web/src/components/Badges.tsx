import type { AnalysisResult } from "../types/api";

function Badge({
  tone,
  children,
  title,
}: {
  tone: "warn" | "info" | "ok";
  children: React.ReactNode;
  title?: string;
}) {
  const style = {
    warn: "bg-amber-500/15 text-amber-200 ring-amber-500/30",
    info: "bg-sky-500/15 text-sky-200 ring-sky-500/30",
    ok: "bg-emerald-500/15 text-emerald-200 ring-emerald-500/30",
  }[tone];
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ring-1 ${style}`}
    >
      {children}
    </span>
  );
}

/**
 * The honesty flags. Each one corresponds to a specific limitation the
 * system knows about itself rather than generic UI chrome.
 */
export function Badges({ result }: { result: AnalysisResult }) {
  return (
    <div className="flex flex-wrap gap-2">
      {result.ood_flag && (
        <Badge
          tone="warn"
          title="The input's characteristics are unlike the model's training distribution, so its confidence is unreliable here."
        >
          Unlike training data
        </Badge>
      )}
      {result.abstained && <Badge tone="warn">Below confidence threshold</Badge>}
      {!result.has_audio && (
        <Badge tone="info" title="No audio stream, so the voice and lip-sync checks did not run.">
          No audio — visual only
        </Badge>
      )}
      {result.n_faces_found === 0 && <Badge tone="warn">No face detected</Badge>}
      {result.has_audio && result.n_faces_found > 0 && !result.ood_flag && !result.abstained && (
        <Badge tone="ok">All streams ran</Badge>
      )}
      <Badge tone="info" title={`Faces found in ${result.n_faces_found} sampled frames`}>
        {result.n_faces_found} face frames
      </Badge>
    </div>
  );
}
