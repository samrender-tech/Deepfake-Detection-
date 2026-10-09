import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalysisResult } from "../types/api";

/**
 * The audio-visual synchrony curve: distance between what the lips did and
 * what the voice did, per window.
 *
 * This is the project's central idea made visible. In most
 * fake videos the face and the voice come from separate tools, so the two
 * drift apart. A flat low curve is consistent; a high or erratic one is not.
 */
export function SyncChart({ result }: { result: AnalysisResult }) {
  if (!result.has_audio) {
    return (
      <section className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
        <h3 className="text-sm font-medium text-neutral-200">Lip-sync agreement</h3>
        <p className="mt-2 text-xs text-neutral-400">
          This clip has no audio track, so the lip-sync check could not run. The
          verdict above is based on the picture alone.
        </p>
      </section>
    );
  }
  if (!result.sync_curve.length) return null;

  const data = result.sync_curve.map((d, i) => ({ w: i, d }));
  const mean = result.sync_curve.reduce((a, b) => a + b, 0) / result.sync_curve.length;

  return (
    <section aria-labelledby="sync-heading" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <h3 id="sync-heading" className="text-sm font-medium text-neutral-200">
        Lip-sync agreement
      </h3>
      <p className="mt-0.5 text-xs text-neutral-400">
        Distance between the lip motion and the speech, per window. Lower is better
        agreement. Average {mean.toFixed(3)}.
      </p>
      <div className="mt-3 h-36">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <defs>
              <linearGradient id="syncFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#27272a" vertical={false} />
            <XAxis dataKey="w" stroke="#52525b" fontSize={11} tickFormatter={(v) => `w${v}`} />
            <YAxis stroke="#52525b" fontSize={11} />
            <Tooltip
              contentStyle={{
                background: "#18181b",
                border: "1px solid #3f3f46",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(v: number) => [v.toFixed(4), "distance"]}
              labelFormatter={(w: number) => `window ${w}`}
            />
            <Area type="monotone" dataKey="d" stroke="#38bdf8" strokeWidth={2} fill="url(#syncFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
