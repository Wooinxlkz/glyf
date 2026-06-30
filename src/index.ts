// ─── GLYF — Public API ───────────────────────────────────────────────────────
// Signature biometric authentication library.
// Novel contributions:
//   1. Multi-channel additive DTW (spatial + velocity + curvature + direction)
//   2. Per-user adaptive Sakoe-Chiba band (derived from enrollment velocity variance)
//   3. Speed-invariant microtremor index (CV of windowed velocity — forgery smoother)
//   4. Inter-stroke rhythm fingerprinting (pause timing as biometric channel)
//   5. Rhythmic autocorrelation (periodic rhythm structure)
//   6. Angular momentum features (rotational pen dynamics)
//   7. Stroke topology fingerprint (scale-invariant directional skeleton)
//   8. Multiplicative gate (timing channels can hard-reject a timing forgery)
//   9. Adversarial forgery simulation (7 attack types, fully automatic, uses v2 DTW)
//  10. Explainable rejection (per-channel failure reasons)
//  11. Non-invertible template hashing

export type {
  StrokePoint,
  SignatureData,
  FeatureVector,
  RhythmProfile,
  AdversarialReport,
  AdversarialResult,
  RejectionReason,
  ExplainedResult,
  EnrollmentTemplate,
  VerificationResult,
} from "./types";

export { preprocess, normalizeCoords, normalizeTime, addVelocity, downsample } from "./normalizer";

export {
  dtw,
  dtwSimilarity,
  compareSequences,
  computeAdaptiveBand,
  computeAdaptiveThreshold,
} from "./dtw";

export { curvatureDtw, curvatureDtwSimilarity } from "./curvdtw";

export {
  extractFeatures,
  featureSimilarity,
  averageFeatureVectors,
  getFeatureWeights,
} from "./features";

export { extractRhythm, averageRhythmProfiles, rhythmSimilarity } from "./rhythm";

export { extractAutocorr, autocorrSimilarity } from "./autocorr";
export type { RhythmAutocorr } from "./autocorr";

export {
  extractAngular,
  angularSimilarity,
  averageAngularProfiles,
} from "./angular";
export type { AngularProfile } from "./angular";

export {
  extractTopology,
  topologySimilarity,
  averageTopologyProfiles,
} from "./topology";
export type { TopologyProfile, StrokeTopology } from "./topology";

export { runAdversarialSimulation } from "./adversarial";
export { explainResult } from "./explainer";
export { hashTemplate, hashTemplateSync, safeStorageTemplate } from "./hasher";

// ─── Channel weights ──────────────────────────────────────────────────────────
// Shape (curvature-weighted DTW): 38%
// Features (12-dim vector):       25%
// Rhythm (histogram + autocorr):  17%   (rhythm 12% + autocorr 5%)
// Angular momentum:               12%
// Stroke topology:                 8%
export const CHANNEL_WEIGHTS = {
  shape: 0.38,
  features: 0.25,
  rhythm: 0.12,
  autocorr: 0.05,
  angular: 0.12,
  topology: 0.08,
} as const;

/** Threshold tiers */
export const THRESHOLDS = {
  STRICT: 82,
  LOOSE: 50,
} as const;

export const MIN_ENROLLMENT_SAMPLES = 3;

// ─── Imports for high-level API ───────────────────────────────────────────────
import type { SignatureData, EnrollmentTemplate, VerificationResult } from "./types";
import { preprocess } from "./normalizer";
import { computeAdaptiveBand, computeAdaptiveThreshold } from "./dtw";
import { curvatureDtw, curvatureDtwSimilarity } from "./curvdtw";
import { extractFeatures, averageFeatureVectors, featureSimilarity } from "./features";
import { extractRhythm, averageRhythmProfiles, rhythmSimilarity } from "./rhythm";
import { extractAutocorr, autocorrSimilarity } from "./autocorr";
import { extractAngular, averageAngularProfiles, angularSimilarity } from "./angular";
import { extractTopology, averageTopologyProfiles, topologySimilarity } from "./topology";
import { explainResult } from "./explainer";
import { hashTemplateSync } from "./hasher";

/**
 * Enroll a user from N signature samples (minimum 3 recommended).
 * Computes all biometric channels and returns a template ready for verification.
 */
export function enroll(samples: SignatureData[]): EnrollmentTemplate {
  if (samples.length < 2) {
    throw new Error(`Need at least 2 samples to enroll, got ${samples.length}`);
  }

  const processedSamples = samples.map((s) => preprocess(s.flatPoints, s));
  const featureVectors = samples.map(extractFeatures);
  const rhythmProfiles = samples.map(extractRhythm);
  const angularProfiles = samples.map(extractAngular);
  const topologyProfiles = samples.map(extractTopology);
  const autocorrProfiles = rhythmProfiles.map(extractAutocorr);

  const adaptiveBand = computeAdaptiveBand(processedSamples);
  const adaptiveThreshold = computeAdaptiveThreshold(processedSamples, adaptiveBand);
  const avgFeatures = averageFeatureVectors(featureVectors);
  const avgRhythm = averageRhythmProfiles(rhythmProfiles);
  const avgAngular = averageAngularProfiles(angularProfiles);
  const avgTopology = averageTopologyProfiles(topologyProfiles);
  // Use first autocorr profile as reference (representative)
  const avgAutocorr = autocorrProfiles[0];
  const templateHash = hashTemplateSync(avgFeatures);

  return {
    rawSamples: samples,
    avgFeatures,
    avgRhythm,
    avgAngular,
    avgTopology,
    avgAutocorr,
    processedSamples,
    adaptiveBandFraction: adaptiveBand,
    adaptiveThreshold,
    templateHash,
    enrolledAt: Date.now(),
    sampleCount: samples.length,
  };
}

/**
 * Verify a test signature against an enrolled template.
 * Runs all 6 biometric channels and returns a weighted combined score.
 */
export function verify(
  template: EnrollmentTemplate,
  test: SignatureData,
  attemptNumber: number = 1
): VerificationResult {
  const testProcessed = preprocess(test.flatPoints, test);
  const testFeatures = extractFeatures(test);
  const testRhythm = extractRhythm(test);
  const testAngular = extractAngular(test);
  const testTopology = extractTopology(test);
  const testAutocorr = extractAutocorr(testRhythm);

  const refProcessed = template.processedSamples[0];
  const band = template.adaptiveBandFraction;

  // Compute per-channel scores
  const rawCurvDist = curvatureDtw(refProcessed, testProcessed, band);
  const shapeScore = curvatureDtwSimilarity(rawCurvDist, refProcessed.length, testProcessed.length);
  const featScore = featureSimilarity(template.avgFeatures, testFeatures);
  const rhythmScore = rhythmSimilarity(template.avgRhythm, testRhythm);
  const autocorrScore = autocorrSimilarity(template.avgAutocorr, testAutocorr);
  const angularScore = angularSimilarity(template.avgAngular, testAngular);
  const topologyScore = topologySimilarity(template.avgTopology, testTopology);

  const w = CHANNEL_WEIGHTS;
  const rawCombined =
    shapeScore    * w.shape    +
    featScore     * w.features +
    rhythmScore   * w.rhythm   +
    autocorrScore * w.autocorr +
    angularScore  * w.angular  +
    topologyScore * w.topology;

  // ── Multiplicative gate on timing-sensitive channels ──────────────────────
  // A forgery that matches shape well but fails rhythm/autocorr is a timing
  // forgery. These channels provide INDEPENDENT biometric evidence — when they
  // fail hard, no amount of shape-match score should rescue the combined score.
  //
  // Rhythm gate mapping (rhythmScore → gate factor):
  //   rhythmScore = 0   → gate = 0.68   (max 32% penalty)
  //   rhythmScore = 65  → gate = 1.00   (no penalty)
  //   rhythmScore > 65  → gate = 1.00   (no penalty)
  //   Single-stroke sigs: rhythmScore defaults to 100 → gate = 1.0 (unaffected)
  //
  // Autocorr gate (secondary, lighter — activates only below 35):
  //   autocorrScore = 0  → gate = 0.88
  //   autocorrScore = 35 → gate = 1.00
  const rhythmGate = rhythmScore < 65
    ? Math.max(0.68, 0.68 + (rhythmScore / 65) * 0.32)
    : 1.0;
  const autocorrGate = autocorrScore < 35
    ? Math.max(0.88, 0.88 + (autocorrScore / 35) * 0.12)
    : 1.0;
  const gate = rhythmGate * autocorrGate;
  const combinedScore = Math.round(rawCombined * gate * 10) / 10;

  const explained = explainResult({
    shapeScore,
    featureScore: featScore,
    rhythmScore,
    angularScore,
    topologyScore,
    combinedScore,
    refFeatures: template.avgFeatures,
    testFeatures,
    refRhythm: template.avgRhythm,
    testRhythm,
    threshold: template.adaptiveThreshold,
    strictThreshold: THRESHOLDS.STRICT,
    looseThreshold: THRESHOLDS.LOOSE,
  });

  const durationMs = test.flatPoints.length > 1
    ? test.flatPoints[test.flatPoints.length - 1].t - test.flatPoints[0].t
    : 0;

  return {
    attempt: attemptNumber,
    shapeScore,
    featureScore: featScore,
    rhythmScore,
    angularScore,
    topologyScore,
    combinedScore,
    durationMs,
    strokePoints: test.flatPoints.length,
    matchStrict: combinedScore >= THRESHOLDS.STRICT,
    matchAdaptive: combinedScore >= template.adaptiveThreshold,
    matchLoose: combinedScore >= THRESHOLDS.LOOSE,
    explained,
    timestamp: Date.now(),
  };
}
