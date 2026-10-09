import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisResult } from "../types/api";

const AXIS = { stroke: "#52525b", fontSize: 11 };

/**
 * Per-frame manipulation score over time.
 *
 * This is the single most useful plot for a user: a forgery is often visible
 * in only a handful of frames (one bad blend on a head turn), and the
 * video-level number averages that away. Clicking a point selects that frame's
 * Grad-CAM overlay.
 */
export function FrameTimeline({
  result,
  onSelectFrame,
  selected,
}: {
  result: AnalysisResult;
  onSelectFrame?: (index: number) => void;
  selected?: number | null;
}) {
  if (!result.frame_scores.length) return null;

  const data = result.frame_scores.map((score, i) => ({
    i,
    t: result.frame_timestamps[i] ?? i,
    score,
  }));
  const segments = result.suspicious_segments ?? [];
  const lastT = data[data.length - 1].t;

  return (
    <section aria-labelledby="timeline-heading" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <h3 id="timeline-heading" className="text-sm font-medium text-neutral-200">
        Per-frame score
      </h3>
      <p className="mt-0.5 text-xs text-neutral-400">
        Higher means that frame looked more manipulated. Click a point to see where the
        model was looking.
      </p>
      <div className="mt-3 h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 4, right: 8, bottom: 4, left: -18 }}
            onClick={(e) => {
              const idx = e?.activeTooltipIndex;
              if (typeof idx === "number" && onSelectFrame) onSelectFrame(idx);
            }}
          >
            <CartesianGrid stroke="#27272a" vertical={false} />
            {segments.map((s, k) => (
              <ReferenceArea
                key={k}
                x1={s.start_s}
                x2={Math.min(s.end_s, lastT)}
                fill="#f43f5e"
                fillOpacity={0.12}
                ifOverflow="extendDomain"
              />
            ))}
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              {...AXIS}
              tickFormatter={(v: number) => `${v.toFixed(1)}s`}
              label={{ value: "time", position: "insideBottomRight", fill: "#52525b", fontSize: 10 }}
            />
            <YAxis domain={[0, 1]} {...AXIS} ticks={[0, 0.5, 1]} />
            <ReferenceLine
              y={result.decision_threshold}
              stroke="#71717a"
              strokeDasharray="3 3"
              label={{ value: "threshold", fill: "#71717a", fontSize: 10, position: "right" }}
            />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #3f3f46",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(v: number) => [v.toFixed(3), "score"]}
              labelFormatter={(t: number) => `at ${Number(t).toFixed(2)}s`}
            />
            <Line
              type="monotone"
              dataKey="score"
              stroke="#a78bfa"
              strokeWidth={2}
              dot={{ r: 2.5, fill: "#a78bfa" }}
              activeDot={{ r: 5 }}
            />
            {selected != null && data[selected] && (
              <ReferenceLine x={data[selected].t} stroke="#f472b6" strokeWidth={1.5} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {segments.length > 0 ? (
        <div className="mt-3" data-testid="segments">
          <p className="text-xs text-neutral-400">
            Flagged segments (frames above the threshold line):
          </p>
          <ul className="mt-1.5 flex flex-wrap gap-1.5">
            {segments.map((s, k) => {
              const peak = data.findIndex((d) => d.t === s.peak_time_s);
              return (
                <li key={k}>
                  <button
                    onClick={() => peak >= 0 && onSelectFrame?.(peak)}
                    className="rounded-full bg-rose-500/10 px-2.5 py-1 text-[11px] tabular-nums text-rose-200 ring-1 ring-rose-500/30 hover:bg-rose-500/20"
                    title={`peak ${s.peak_score.toFixed(3)} at ${s.peak_time_s.toFixed(2)}s`}
                  >
                    {s.start_s.toFixed(1)}s–{s.end_s.toFixed(1)}s · {s.n_frames}{" "}
                    {s.n_frames === 1 ? "frame" : "frames"}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : (
        <p className="mt-3 text-xs text-neutral-500">
          No frame scored above the threshold line.
        </p>
      )}
    </section>
  );
}
