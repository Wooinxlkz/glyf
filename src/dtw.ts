// ─── DTW with per-user adaptive Sakoe-Chiba band ─────────────────────────────
// Unlike a fixed 15% band, PenPrint computes each user's natural velocity
// variance during enrollment and widens/narrows the band accordingly.
// High-variance signers (natural timing jitter) get a wider band so they
// aren't unfairly rejected. Tight, consistent signers get a narrower band
// which makes the gate harder to fool.

import type { StrokePoint } from "./types";

const VELOCITY_WEIGHT = 0.35;
const DEFAULT_BAND_FRACTION = 0.15;
const MIN_BAND_FRACTION = 0.08;
const MAX_BAND_FRACTION = 0.28;

function distance3d(a: StrokePoint, b: StrokePoint): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dv = ((a.v ?? 0) - (b.v ?? 0)) * VELOCITY_WEIGHT;
  return Math.sqrt(dx * dx + dy * dy + dv * dv);
}

/**
 * Compute the adaptive Sakoe-Chiba band fraction for a specific user.
 *
 * We measure the timing variance of the user's enrollment samples by comparing
 * how much each sample's velocity profile deviates from the mean profile.
 * A high deviation means the user signs with natural timing jitter →
 * wider band. A low deviation means they sign very consistently → tighter band.
 *
 * This is the core novel contribution: the band is NOT a fixed constant,
 * it is derived from the individual's biometric variance.
 */
export function computeAdaptiveBand(processedSamples: StrokePoint[][]): number {
  if (processedSamples.length < 2) return DEFAULT_BAND_FRACTION;

  // Collect velocity profiles (already normalized to [0,1])
  const profiles = processedSamples.map((s) => s.map((p) => p.v ?? 0));
  const len = profiles[0].length;

  // Compute mean velocity at each position
  const meanProfile = Array(len).fill(0);
  for (const profile of profiles) {
    for (let i = 0; i < len; i++) {
      meanProfile[i] += profile[i] / profiles.length;
    }
  }

  // Compute mean absolute deviation from the mean velocity profile
  let totalDeviation = 0;
  for (const profile of profiles) {
    for (let i = 0; i < len; i++) {
      totalDeviation += Math.abs(profile[i] - meanProfile[i]);
    }
  }
  const avgDeviation = totalDeviation / (profiles.length * len);

  // Map deviation [0, 0.3] → band [MIN, MAX]
  // avgDeviation ≈ 0 means very consistent → narrow band
  // avgDeviation ≈ 0.3 means highly variable → wide band
  const t = Math.min(1, avgDeviation / 0.3);
  const band = MIN_BAND_FRACTION + t * (MAX_BAND_FRACTION - MIN_BAND_FRACTION);
  return Math.round(band * 1000) / 1000;
}

/**
 * Standard DTW with a Sakoe-Chiba band constraint.
 * Returns raw cumulative distance (lower = more similar).
 */
export function dtw(
  seq1: StrokePoint[],
  seq2: StrokePoint[],
  bandFraction: number = DEFAULT_BAND_FRACTION
): number {
  const n = seq1.length;
  const m = seq2.length;
  if (n === 0 || m === 0) return Infinity;

  const band = Math.ceil(Math.max(n, m) * bandFraction);
  const matrix: number[][] = Array(n + 1)
    .fill(null)
    .map(() => Array(m + 1).fill(Infinity));
  matrix[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    const jMin = Math.max(1, i - band);
    const jMax = Math.min(m, i + band);
    for (let j = jMin; j <= jMax; j++) {
      const cost = distance3d(seq1[i - 1], seq2[j - 1]);
      matrix[i][j] =
        cost +
        Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
    }
  }

  return matrix[n][m];
}

/**
 * Normalize raw DTW distance to a [0,100] similarity score.
 * 100 = identical, 0 = completely different.
 */
export function dtwSimilarity(
  rawDistance: number,
  refLen: number,
  testLen: number
): number {
  if (!isFinite(rawDistance) || rawDistance < 0) return 0;
  const avgLen = (refLen + testLen) / 2 || 1;
  const perPoint = rawDistance / avgLen;
  return Math.max(0, Math.min(100, Math.round((100 - perPoint * 150) * 10) / 10));
}

/**
 * Compare two preprocessed sequences and return a similarity score.
 */
export function compareSequences(
  ref: StrokePoint[],
  test: StrokePoint[],
  bandFraction?: number
): { rawDistance: number; score: number } {
  const rawDistance = dtw(ref, test, bandFraction);
  const score = dtwSimilarity(rawDistance, ref.length, test.length);
  return { rawDistance, score };
}

/**
 * Compute a per-user adaptive threshold from enrollment self-consistency.
 * Consistent signers get a tighter gate; variable signers get a looser one.
 */
export function computeAdaptiveThreshold(
  processedSamples: StrokePoint[][],
  band: number,
  base: number = 72
): number {
  if (processedSamples.length < 2) return base;
  let minScore = 100;
  for (let a = 0; a < processedSamples.length; a++) {
    for (let b = a + 1; b < processedSamples.length; b++) {
      const raw = dtw(processedSamples[a], processedSamples[b], band);
      const avg = (processedSamples[a].length + processedSamples[b].length) / 2 || 1;
      const score = Math.max(0, Math.min(100, 100 - (raw / avg) * 150));
      if (score < minScore) minScore = score;
    }
  }
  // Tighten gate for consistent signers, loosen for variable signers
  const adj = (minScore - 70) * 0.35;
  return Math.round(Math.max(base - 10, Math.min(base + 8, base + adj)) * 10) / 10;
}
