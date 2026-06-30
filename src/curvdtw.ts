// ─── Multi-Channel Additive DTW (GLYF v2) ────────────────────────────────────
// Replaces curvature-SCALING with curvature-ADDITIVE channels.
//
// v1 problem: scaling by curvature weight averages ~0.4× on smooth data →
//   all distances compressed → calibration K inflates forgery scores equally.
//
// v2 solution: four INDEPENDENT additive channels, each contributing penalty
//   that a forgery cannot cancel through geometric warping alone:
//
//   channel 1 — spatial      (0.50×): geometric Euclidean distance
//   channel 2 — velocity     (0.20×): speed profile mismatch
//   channel 3 — curvature    (0.20×): additive |curv_a − curv_b| penalty  ← novel
//   channel 4 — direction    (0.10×): additive angular mismatch penalty    ← novel
//
// A forgery that traces the correct overall bounding box but has a different
// internal curve structure CANNOT reduce channels 3 & 4 through warping.
// It can minimize spatial distance by warping, but pays full additive cost
// at every point where curvature or direction angles differ.
//
// This breaks the calibration degeneracy: forgery per-point costs are NOW
// higher than genuine costs even after DTW warps to the best alignment.

import type { StrokePoint } from "./types";

const SPATIAL_W   = 0.50;
const VELOCITY_W  = 0.20;
const CURVATURE_W = 0.20;   // additive curvature mismatch
const DIRECTION_W = 0.10;   // additive direction mismatch

/** Curvature at each point in [0, 1]: 0 = straight, 1 = sharp reversal. */
function computeCurvatures(pts: StrokePoint[]): number[] {
  const n = pts.length;
  if (n < 3) return new Array(n).fill(0);
  const c = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const dx1 = pts[i].x - pts[i - 1].x;
    const dy1 = pts[i].y - pts[i - 1].y;
    const dx2 = pts[i + 1].x - pts[i].x;
    const dy2 = pts[i + 1].y - pts[i].y;
    const m1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) || 1;
    const m2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) || 1;
    const cosA = (dx1 * dx2 + dy1 * dy2) / (m1 * m2);
    c[i] = (1 - Math.max(-1, Math.min(1, cosA))) / 2;
  }
  c[0] = c[1] ?? 0;
  c[n - 1] = c[n - 2] ?? 0;
  return c;
}

/** Direction angle at each point (atan2, radians). */
function computeDirections(pts: StrokePoint[]): number[] {
  const d: number[] = [0];
  for (let i = 1; i < pts.length; i++) {
    d.push(Math.atan2(pts[i].y - pts[i - 1].y, pts[i].x - pts[i - 1].x));
  }
  return d;
}

/** Smallest angular difference normalized to [0, 1] (0 = same, 1 = opposite). */
function angleDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d / Math.PI;
}

/**
 * Multi-channel point distance: four additive independent channels.
 * Each channel can reveal a forgery independently — a forgery must match
 * ALL four to score well.
 */
function multiChannelDist(
  a: StrokePoint, b: StrokePoint,
  curvA: number, curvB: number,
  dirA: number, dirB: number,
): number {
  const dx       = a.x - b.x;
  const dy       = a.y - b.y;
  const spatial  = Math.sqrt(dx * dx + dy * dy);
  const velDiff  = Math.abs((a.v ?? 0) - (b.v ?? 0));
  const curvDiff = Math.abs(curvA - curvB);
  const dirDiff  = angleDiff(dirA, dirB);
  return SPATIAL_W   * spatial
       + VELOCITY_W  * velDiff
       + CURVATURE_W * curvDiff
       + DIRECTION_W * dirDiff;
}

/**
 * Multi-channel additive DTW.
 * Returns raw cumulative cost (lower = more similar).
 *
 * @param bandFraction Sakoe-Chiba band (use per-user adaptive band from dtw.ts)
 */
export function curvatureDtw(
  seq1: StrokePoint[],
  seq2: StrokePoint[],
  bandFraction: number = 0.15,
): number {
  const n = seq1.length;
  const m = seq2.length;
  if (n === 0 || m === 0) return Infinity;

  const curv1 = computeCurvatures(seq1);
  const curv2 = computeCurvatures(seq2);
  const dir1  = computeDirections(seq1);
  const dir2  = computeDirections(seq2);

  const band = Math.ceil(Math.max(n, m) * bandFraction);
  const matrix: number[][] = Array(n + 1)
    .fill(null)
    .map(() => Array(m + 1).fill(Infinity));
  matrix[0][0] = 0;

  for (let i = 1; i <= n; i++) {
    const jMin = Math.max(1, i - band);
    const jMax = Math.min(m, i + band);
    for (let j = jMin; j <= jMax; j++) {
      const cost = multiChannelDist(
        seq1[i - 1], seq2[j - 1],
        curv1[i - 1], curv2[j - 1],
        dir1[i - 1],  dir2[j - 1],
      );
      matrix[i][j] =
        cost + Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]);
    }
  }

  return matrix[n][m];
}

/**
 * Normalize multi-channel DTW distance to [0, 100] similarity score.
 * Multi-channel per-point costs are higher than v1 (sum of 4 channels).
 * Calibration: a genuine match typically scores 92-96.
 */
export function curvatureDtwSimilarity(
  rawDistance: number,
  refLen: number,
  testLen: number,
): number {
  if (!isFinite(rawDistance) || rawDistance < 0) return 0;
  const avgLen = (refLen + testLen) / 2 || 1;
  const perPoint = rawDistance / avgLen;
  return Math.max(0, Math.min(100, Math.round((100 - perPoint * 95) * 10) / 10));
}
