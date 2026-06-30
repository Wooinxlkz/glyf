// ─── Core data types for GLYF ────────────────────────────────────────────────

export interface StrokePoint {
  x: number;
  y: number;
  t: number;
  v?: number;        // normalized velocity [0,1], added during preprocessing
  pressure?: number; // stylus pressure [0,1] if device supports it
}

export interface SignatureData {
  strokes: StrokePoint[][];
  flatPoints: StrokePoint[];
  capturedAt: number;
  devicePixelRatio?: number; // for device-agnostic normalization
  canvasWidth?: number;
  canvasHeight?: number;
}

export interface FeatureVector {
  // Geometric
  strokeCount: number;
  aspectRatio: number;
  totalPathLength: number;
  // Dynamic
  avgVelocity: number;
  velocityVariance: number;
  peakVelocity: number;
  avgAcceleration: number;
  terminalVelocityProfile: number;
  // Structural
  curvatureEntropy: number;
  directionChangeRate: number;
  // Temporal
  durationMs: number;
  strokeDurationVariance: number;
  // GLYF v2 novel channels
  microtremorIndex: number;       // high-freq velocity variance (muscle tremor fingerprint)
  interStrokeRhythmRatio: number; // mean inter-stroke pause / total duration
}

export interface RhythmProfile {
  pauseDurations: number[];
  pauseRatios: number[];
  rhythmVariance: number;
  rhythmSignature: number[];
}

export interface AdversarialResult {
  forgeryType: string;
  score: number;
  passed: boolean;
}

export interface AdversarialReport {
  simulatedFAR: number;
  worstCaseScore: number;
  totalAttempts: number;
  passedAttempts: number;
  results: AdversarialResult[];
  verdict: "SECURE" | "MARGINAL" | "VULNERABLE";
}

export interface RejectionReason {
  channel: "shape" | "features" | "rhythm" | "angular" | "topology";
  field: string;
  expected: number;
  actual: number;
  deviation: number;
  message: string;
}

export interface ExplainedResult {
  passed: boolean;
  // Individual channel scores
  shapeScore: number;       // multi-channel additive DTW
  featureScore: number;
  rhythmScore: number;      // histogram + autocorrelation
  angularScore: number;     // angular momentum
  topologyScore: number;    // stroke topology fingerprint
  combinedScore: number;
  threshold: number;
  topReasons: RejectionReason[];
  label: string;
}

export interface EnrollmentTemplate {
  rawSamples: SignatureData[];
  avgFeatures: FeatureVector;
  avgRhythm: RhythmProfile;
  processedSamples: StrokePoint[][];
  adaptiveBandFraction: number;
  adaptiveThreshold: number;
  templateHash: string;
  enrolledAt: number;
  sampleCount: number;
  // Novel channel profiles stored at enrollment
  avgAngular: import("./angular").AngularProfile;
  avgTopology: import("./topology").TopologyProfile;
  avgAutocorr: import("./autocorr").RhythmAutocorr;
}

export interface VerificationResult {
  attempt: number;
  shapeScore: number;
  featureScore: number;
  rhythmScore: number;
  angularScore: number;
  topologyScore: number;
  combinedScore: number;
  durationMs: number;
  strokePoints: number;
  matchStrict: boolean;
  matchAdaptive: boolean;
  matchLoose: boolean;
  explained: ExplainedResult;
  timestamp: number;
}
