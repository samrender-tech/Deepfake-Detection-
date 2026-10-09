import { useCallback, useRef, useState } from "react";
import type { Limits } from "../types/api";

/** The API's default rate limit is 10 analyses per minute per client. */
export const MAX_QUEUE = 10;

/**
 * Drag-and-drop upload with client-side pre-checks.
 *
 * The checks here duplicate the server's deliberately: catching an oversized
 * or overlong clip before a 100 MB upload is a courtesy, not a security
 * boundary. The server re-validates everything (api/security.py).
 */
export function Upload({
  limits,
  disabled,
  onFile,
  onFiles,
}: {
  limits: Limits | null;
  disabled?: boolean;
  onFile: (f: File) => void;
  /** When set, selecting several files queues them all instead of the first. */
  onFiles?: (fs: File[]) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const check = useCallback(
    (file: File): string | null => {
      if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|mkv|avi|flv)$/i.test(file.name)) {
        return "That does not look like a video file.";
      }
      const maxMb = limits?.max_upload_mb ?? 100;
      if (file.size > maxMb * 1024 * 1024) {
        return `That file is ${(file.size / 1024 / 1024).toFixed(0)} MB; the limit is ${maxMb} MB.`;
      }
      return null;
    },
    [limits],
  );

  const accept = useCallback(
    (list: FileList | null | undefined) => {
      setLocalError(null);
      const files = Array.from(list ?? []);
      if (!files.length) return;

      if (files.length === 1 || !onFiles) {
        const err = check(files[0]);
        if (err) setLocalError(err);
        else onFile(files[0]);
        return;
      }

      // Several files: queue the good ones, and say which were skipped and why
      // rather than dropping them silently.
      const good: File[] = [];
      const skipped: string[] = [];
      for (const f of files) {
        const err = check(f);
        if (err) skipped.push(`${f.name}: ${err}`);
        else good.push(f);
      }
      const queued = good.slice(0, MAX_QUEUE);
      if (good.length > MAX_QUEUE) {
        skipped.push(`${good.length - MAX_QUEUE} more file(s): at most ${MAX_QUEUE} at a time.`);
      }
      if (skipped.length) setLocalError(`Skipped — ${skipped.join(" · ")}`);
      if (queued.length) onFiles(queued);
    },
    [check, onFile, onFiles],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-disabled={disabled}
        aria-label="Choose or drop a video file to analyse"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files);
        }}
        className={`flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition
          ${dragging ? "border-violet-400 bg-violet-400/5" : "border-neutral-700 hover:border-neutral-500"}
          ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <p className="text-sm font-medium text-neutral-200">
          {onFiles
            ? "Drop videos here, or press to choose — one or several"
            : "Drop a video here, or press to choose one"}
        </p>
        <p className="mt-1.5 text-xs text-neutral-500">
          {limits
            ? `Up to ${limits.max_upload_mb} MB and ${limits.max_duration_s}s · ${limits.accepted_containers.join(", ")}`
            : "MP4, MOV, WebM, MKV, AVI"}
        </p>
        {limits && (
          <p className="mt-3 max-w-md text-[11px] leading-relaxed text-neutral-600">
            Your upload is deleted as soon as it has been analysed, and the result
            is removed after {limits.retention_hours} hours.
          </p>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mp4,.mov,.webm,.mkv,.avi,.flv"
        className="sr-only"
        multiple={Boolean(onFiles)}
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />

      {localError && (
        <p role="alert" className="mt-3 rounded-md bg-amber-500/10 p-2.5 text-xs text-amber-200">
          {localError}
        </p>
      )}
    </div>
  );
}
