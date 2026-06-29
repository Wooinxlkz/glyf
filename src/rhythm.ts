// ─── Inter-stroke rhythm fingerprinting ──────────────────────────────────────
// The pause timing between strokes is a biometric channel separate from shape.
// How long a signer pauses before each new stroke is as distinctive as how
// they draw it — and much harder to consciously replicate under pressure.
//
// This is one of PenPrint's novel contributions: treating inter-stroke rhythm
// as an independent channel with its own similarity score rather than folding
// pause timing into the general feature vector.

import type { SignatureData, RhythmProfile } from "./types";

const RHYTHM_BINS = 8;

/**
 * Extract inter-stroke pause durations from raw stroke data.
 * A "pause" is the gap between the last point of stroke N and the
 * first point of stroke N+1.
 */
export function extractRhythm(sig: SignatureData): RhythmProfile {
  const strokes = sig.strokes;
  if (strokes.length < 2) {
    return {
      pauseDurations: [],
      pauseRatios: [],
      rhythmVariance: 0,
      rhythmSignature: new Array(RHYTHM_BINS).fill(0),
    };
  }

  const totalDuration = sig.flatPoints.length > 1
    ? sig.flatPoints[sig.flatPoints.length - 1].t - sig.flatPoints[0].t
    : 1;

  const pauseDurations: number[] = [];
  for (let i = 0; i < strokes.length - 1; i++) {
    const strokeEnd = strokes[i][strokes[i].length - 1];
    const strokeStart = strokes[i + 1][0];
    if (strokeEnd && strokeStart) {
      pauseDurations.push(Math.max(0, strokeStart.t - strokeEnd.t));
    }
  }

  if (pauseDurations.length === 0) {
    return {
      pauseDurations: [],
      pauseRatios: [],
      rhythmVariance: 0,
      rhythmSignature: new Array(RHYTHM_BINS).fill(0),
    };
  }

  // Normalize pauses by total signing duration → device/speed agnostic
  const pauseRatios = pauseDurations.map((d) => d / totalDuration);

  // Variance of pause ratios — lower = more rhythmically consistent signer
  const mean = pauseRatios.reduce((a, b) => a + b, 0) / pauseRatios.length;
  const rhythmVariance =
    pauseRatios.reduce((a, b) => a + (b - mean) ** 2, 0) / pauseRatios.length;

  // 8-bin histogram of pause ratio distribution (rhythm fingerprint)
  // Each bin covers a range of [0, 0.5] total-duration units
  const rhythmSignature = new Array(RHYTHM_BINS).fill(0);
  for (const r of pauseRatios) {
    const bin = Math.min(RHYTHM_BINS - 1, Math.floor((r / 0.5) * RHYTHM_BINS));
    rhythmSignature[bin]++;
  }
  // Normalize histogram to [0,1]
  const total = pauseRatios.length || 1;
  for (let i = 0; i < RHYTHM_BINS; i++) {
    rhythmSignature[i] /= total;
  }

  return { pauseDurations, pauseRatios, rhythmVariance, rhythmSignature };
}

/**
 * Average multiple rhythm profiles into one reference profile.
 */
export function averageRhythmProfiles(profiles: RhythmProfile[]): RhythmProfile {
  const valid = profiles.filter((p) => p.pauseRatios.length > 0);
  if (valid.length === 0) {
    return {
      pauseDurations: [],
      pauseRatios: [],
      rhythmVariance: 0,
      rhythmSignature: new Array(RHYTHM_BINS).fill(0),
    };
  }

  // Average the rhythm signature histograms
  const avgSig = new Array(RHYTHM_BINS).fill(0);
  for (const p of valid) {
    for (let i = 0; i < RHYTHM_BINS; i++) {
      avgSig[i] += p.rhythmSignature[i] / valid.length;
    }
  }

  const avgVariance = valid.reduce((s, p) => s + p.rhythmVariance, 0) / valid.length;

  return {
    pauseDurations: valid[0].pauseDurations, // representative sample
    pauseRatios: valid[0].pauseRatios,
    rhythmVariance: avgVariance,
    rhythmSignature: avgSig,
  };
}

/**
 * Compare a test rhythm profile against the reference.
 * Returns a similarity score [0, 100].
 *
 * Two components:
 *   1. Histogram similarity (Bhattacharyya coefficient)
 *   2. Variance similarity (consistent signers penalized more for changes)
 */
export function rhythmSimilarity(ref: RhythmProfile, test: RhythmProfile): number {
  // If either side has no pauses (single-stroke signature), skip rhythm
  if (ref.rhythmSignature.length === 0 || test.rhythmSignature.length === 0) return 100;
  if (ref.pauseRatios.length === 0 || test.pauseRatios.length === 0) return 100;

  // Bhattacharyya coefficient: measures overlap between two histograms
  let bc = 0;
  for (let i = 0; i < RHYTHM_BINS; i++) {
    bc += Math.sqrt(ref.rhythmSignature[i] * test.rhythmSignature[i]);
  }
  const histScore = bc * 100; // bc ∈ [0,1] → [0,100]

  // Variance similarity: penalize large changes in signing rhythm regularity
  const varDiff = Math.abs(ref.rhythmVariance - test.rhythmVariance);
  const varScale = Math.max(ref.rhythmVariance, 0.001);
  const varScore = Math.max(0, 100 - (varDiff / varScale) * 50);

  return Math.round((histScore * 0.7 + varScore * 0.3) * 10) / 10;
}
