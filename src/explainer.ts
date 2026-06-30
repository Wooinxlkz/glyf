// ─── Explainable rejection ────────────────────────────────────────────────────
// When a signature is rejected, GLYF returns the top 3 reasons ranked by
// contribution to failure — in plain language, per channel.

import type { FeatureVector, RhythmProfile, RejectionReason, ExplainedResult } from "./types";

const FEATURE_LABELS: Record<keyof FeatureVector, string> = {
  strokeCount: "Number of strokes",
  aspectRatio: "Width-to-height ratio",
  totalPathLength: "Total path length",
  avgVelocity: "Average signing speed",
  velocityVariance: "Speed consistency",
  peakVelocity: "Peak speed",
  avgAcceleration: "Acceleration profile",
  terminalVelocityProfile: "Pen-lift deceleration pattern",
  curvatureEntropy: "Curve complexity",
  directionChangeRate: "Direction change rate",
  durationMs: "Total signing duration",
  strokeDurationVariance: "Stroke timing variance",
  // GLYF v2 novel channels
  microtremorIndex: "Micro-tremor pattern (CV of velocity — muscle tremor fingerprint)",
  interStrokeRhythmRatio: "Inter-stroke pause timing ratio (pen-lift rhythm)",
};

function featureDeviation(ref: number, test: number): number {
  const scale = Math.max(Math.abs(ref), Math.abs(test), 0.001);
  return Math.abs(ref - test) / scale;
}

function describeDeviation(field: keyof FeatureVector, ref: number, test: number): string {
  const ratio = test / (ref || 0.001);
  switch (field) {
    case "strokeCount":
      return test < ref ? `Missing ${ref - test} stroke(s)` : `${test - ref} extra stroke(s)`;
    case "durationMs":
      return ratio < 0.7 ? "Signed too quickly" : "Signed too slowly";
    case "avgVelocity":
      return ratio > 1.3 ? "Moving too fast" : "Moving too slowly";
    case "aspectRatio":
      return ratio < 0.8 ? "Signature too narrow" : "Signature too wide";
    case "totalPathLength":
      return ratio < 0.75 ? "Path too short (possibly incomplete)" : "Path too long";
    case "curvatureEntropy":
      return test < ref * 0.7 ? "Curves too simple / too straight" : "Too many direction changes";
    case "terminalVelocityProfile":
      return "Pen-lift pattern differs (deceleration does not match)";
    case "avgAcceleration":
      return test > ref * 1.4 ? "Signing too jerkily" : "Signing too smoothly (robot-like)";
    default:
      return `Expected approx ${Math.round(ref * 100) / 100}, got ${Math.round(test * 100) / 100}`;
  }
}

export function explainResult(params: {
  shapeScore: number;
  featureScore: number;
  rhythmScore: number;
  angularScore: number;
  topologyScore: number;
  combinedScore: number;
  refFeatures: FeatureVector;
  testFeatures: FeatureVector;
  refRhythm: RhythmProfile;
  testRhythm: RhythmProfile;
  threshold: number;
  strictThreshold?: number;
  looseThreshold?: number;
}): ExplainedResult {
  const {
    shapeScore, featureScore, rhythmScore, angularScore, topologyScore,
    combinedScore, refFeatures, testFeatures, refRhythm,
    threshold, strictThreshold = 82,
  } = params;

  const passed = combinedScore >= threshold;
  let topReasons: RejectionReason[] = [];

  if (!passed) {
    const reasons: RejectionReason[] = [];

    if (shapeScore < threshold) {
      reasons.push({
        channel: "shape",
        field: "curvatureDtw",
        expected: threshold,
        actual: shapeScore,
        deviation: (threshold - shapeScore) / threshold,
        message: `Shape mismatch — curvature-weighted path score ${shapeScore} < ${threshold}`,
      });
    }

    if (angularScore < 60) {
      reasons.push({
        channel: "angular",
        field: "angularMomentum",
        expected: 80,
        actual: angularScore,
        deviation: (80 - angularScore) / 80,
        message: "Rotational pen dynamics differ (angular momentum mismatch)",
      });
    }

    if (topologyScore < 55) {
      reasons.push({
        channel: "topology",
        field: "strokeSkeleton",
        expected: 75,
        actual: topologyScore,
        deviation: (75 - topologyScore) / 75,
        message: "Stroke direction structure differs (topology mismatch)",
      });
    }

    if (rhythmScore < 60 && refRhythm.pauseRatios.length > 0) {
      reasons.push({
        channel: "rhythm",
        field: "interStrokePauses",
        expected: 80,
        actual: rhythmScore,
        deviation: (80 - rhythmScore) / 80,
        message: "Pause timing between strokes does not match enrolled pattern",
      });
    }

    const keys = Object.keys(refFeatures) as (keyof FeatureVector)[];
    const featReasons = keys
      .map((key) => {
        const ref = refFeatures[key] as number;
        const actual = testFeatures[key] as number;
        const deviation = featureDeviation(ref, actual);
        return {
          channel: "features" as const,
          field: key,
          expected: ref,
          actual,
          deviation,
          message: `${FEATURE_LABELS[key]}: ${describeDeviation(key, ref, actual)}`,
        };
      })
      .sort((a, b) => b.deviation - a.deviation)
      .slice(0, 2);

    reasons.push(...featReasons);
    topReasons = reasons.sort((a, b) => b.deviation - a.deviation).slice(0, 3);
  }

  let label: string;
  if (passed) {
    label = combinedScore >= strictThreshold
      ? "Verified (strict)"
      : combinedScore >= threshold
        ? "Verified (adaptive)"
        : "Verified (loose)";
  } else {
    label = topReasons.length > 0
      ? `Rejected — ${topReasons[0].message}`
      : "Rejected";
  }

  return {
    passed,
    shapeScore,
    featureScore,
    rhythmScore,
    angularScore,
    topologyScore,
    combinedScore,
    threshold,
    topReasons,
    label,
  };
}
