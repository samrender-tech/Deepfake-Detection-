// Mirrors api/schemas.py. Kept hand-written rather than generated so the
// three-state verdict and the required disclaimer are visible in the type
// system -- a client literally cannot type a boolean "isFake".

export type Verdict = "likely_authentic" | "likely_manipulated" | "inconclusive";
export type JobState = "queued" | "running" | "done" | "failed" | "cancelled";
export type Stage =
  | "queued" | "probe" | "preprocess" | "infer"
  | "calibrate" | "explain" | "persist" | "done";

export interface StreamScore {
  name: "visual" | "audio" | "sync";
  score: number | null;
  available: boolean;
  reliability: number;
  note: string;
}

export interface SuspiciousSegment {
  start_s: number;
  end_s: number;
  peak_score: number;
  peak_time_s: number;
  n_frames: number;
}

export interface AnalysisResult {
  verdict: Verdict;
  calibrated_probability: number;
  confidence_interval: [number, number];
  decision_threshold: number;
  abstained: boolean;
  ood_flag: boolean;
  calibration_reliable: boolean;
  warnings: string[];
  disclaimer: string;
  streams: StreamScore[];
  frame_scores: number[];
  frame_timestamps: number[];
  sync_curve: number[];
  suspicious_segments: SuspiciousSegment[];
  gate_weights: Record<string, number>;
  reliability: Record<string, number>;
  has_audio: boolean;
  n_faces_found: number;
  gradcam_urls: Record<string, string>;
  spectrum_url: string | null;
  explanation: string | null;
  explanation_source: "agent" | "template" | "none";
  model_version: string;
  latency_ms: Record<string, number>;
}

export interface AnalysisStatus {
  job_id: string;
  state: JobState;
  stage: Stage;
  progress: number;
  created_at: string;
  updated_at: string;
  filename: string | null;
  error: string | null;
  result: AnalysisResult | null;
}

export interface Limits {
  max_upload_mb: number;
  max_duration_s: number;
  accepted_containers: string[];
  retention_hours: number;
  disclaimer: string;
}

export interface ModelInfo {
  model_version: string;
  architecture: string;
  device: string;
  trained_on: string;
  n_frames: number;
  image_size: number;
  audio_seconds: number;
  calibration_method: string;
  temperature: number;
  decision_threshold: number;
  threshold_criterion: string;
  conformal_band: [number, number];
  conformal_alpha: number;
  calibration_fitted_on: string;
  calibration_n_val: number;
  calibration_reliable: boolean;
  ood_detection: boolean;
  disclaimer: string;
}

export type ReportFormat = "html" | "json";

export const VERDICT_LABEL: Record<Verdict, string> = {
  likely_authentic: "Likely authentic",
  likely_manipulated: "Likely manipulated",
  inconclusive: "Inconclusive",
};

export const STAGE_LABEL: Record<Stage, string> = {
  queued: "Queued",
  probe: "Reading the file",
  preprocess: "Finding and cropping faces",
  infer: "Running the detector",
  calibrate: "Calibrating confidence",
  explain: "Building the explanation",
  persist: "Saving results",
  done: "Done",
};
