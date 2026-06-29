// ─── Curvature-Weighted DTW (GLYF novel primitive) ───────────────────────────
// Standard DTW weights all points equally. GLYF's curvature-weighted DTW
// assigns higher importance to high-curvature points (corners, loops,
// direction reversals) — the parts of a signature that are most distinctive.
// Straight-line segments between strokes are low-information filler;
// curvature events are where identity lives.
//
// Method: compute local curvature at each point as the magnitude of the
// angle change between the incoming and outgoing tangent vectors.
// Use curvature as a per-point multiplier on the distance metric.

import type { StrokePoint } from "./types";

const VELOCITY_WEIGHT = 0.35;
const CURVATURE_BOOST = 2.5; // max weight amplification at high-curvature points
const MIN_WEIGHT = 0.4;      // minimum weight for near-straight segments

/**
 * Compute local curvature at each point in [0, 1].
 * 0 = perfectly straight, 1 = sharp reversal.
 */
function computeCurvatures(pts: StrokePoint[]): number[] {
  const n = pts.length;
  if (n < 3) return new Array(n).fill(0);

  const curvatures = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const dx1 = pts[i].x - pts[i - 1].x;
    const dy1 = pts[i].y - pts[i - 1].y;
    const dx2 = pts[i + 1].x - pts[i].x;
    const dy2 = pts[i + 1].y - pts[i].y;

    const mag1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) || 1;
    const mag2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) || 1;

    // cos(angle) between tangent vectors
    const cosAngle = (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2);
    // curvature = 0 when straight (cosAngle ≈ 1), 1 when reversed (cosAngle ≈ -1)
    curvatures[i] = (1 - Math.max(-1, Math.min(1, cosAngle))) / 2;
  }
  // Endpoints inherit neighbor curvature
  curvatures[0] = curvatures[1];
  curvatures[n - 1] = curvatures[n - 2];
  return curvatures;
}

/**
 * Map curvature [0,1] to a weight multiplier [MIN_WEIGHT, CURVATURE_BOOST].
 */
function curvatureWeight(c: number): number {
  return MIN_WEIGHT + c * (CURVATURE_BOOST - MIN_WEIGHT);
}

/**
 * Curvature-weighted point distance.
 * Distance is scaled by the average curvature weight of both points.
 * High-curvature matches that fail are penalized more; matches at
 * low-curvature (straight) segments matter less.
 */
function weightedDistance(
  a: StrokePoint, b: StrokePoint,
  wA: number, wB: number
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dv = ((a.v ?? 0) - (b.v ?? 0)) * VELOCITY_WEIGHT;
  const euclidean = Math.sqrt(dx * dx + dy * dy + dv * dv);
  return euclidean * (wA + wB) / 2;
}

/**
 * Run curvature-weighted DTW between two preprocessed sequences.
 * Returns raw cumulative cost (lower = more similar).
 *
 * @param bandFraction Sakoe-Chiba band (use per-user adaptive band from dtw.ts)
 */
export function curvatureDtw(
  seq1: StrokePoint[],
  seq2: StrokePoint[],
  bandFraction: number = 0.15
): number {
  const n = seq1.length;
  const m = seq2.length;
  if (n === 0 || m === 0) return Infinity;

  const w1 = computeCurvatures(seq1).map(curvatureWeight);
  const w2 = computeCurvatures(seq2).map(curvatureWeight);

  const band = Math.ceil(Math.max(n, m) * bandFraction);
  const matrix: number[][] = Array(n + 1)
    .fill(null)
    .map(() => Array(m + 1).fill(Infinity));
  matrix[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    const jMin = Math.max(1, i - band);
    const jMax = Math.min(m, i + band);
    for (let j = jMin; j <= jMax; j++) {
      const cost = weightedDistance(seq1[i - 1], seq2[j - 1], w1[i - 1], w2[j - 1]);
      matrix[i][j] =
        cost + Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
    }
  }

  return matrix[n][m];
}

/**
 * Normalize curvature-DTW distance to [0, 100] similarity score.
 */
export function curvatureDtwSimilarity(
  rawDistance: number,
  refLen: number,
  testLen: number
): number {
  if (!isFinite(rawDistance) || rawDistance < 0) return 0;
  const avgLen = (refLen + testLen) / 2 || 1;
  // Curvature-weighted costs are larger on average, so use a slightly
  // looser normalization constant (120 vs 150 in standard DTW).
  const perPoint = rawDistance / avgLen;
  return Math.max(0, Math.min(100, Math.round((100 - perPoint * 120) * 10) / 10));
}
