// ─── Multi-Channel Additive DTW (GLYF v2, 5-channel) ─────────────────────────
//
// v1 problem: scaling by curvature weight averages ~0.4× on smooth data →
//   all distances compressed → calibration K inflates forgery scores equally.
//
// v2 solution: five INDEPENDENT additive channels, each contributing penalty
//   that a forgery cannot cancel through geometric warping alone:
//
//   channel 1 — spatial      (0.42×): geometric Euclidean distance
//   channel 2 — velocity     (0.17×): speed profile mismatch
//   channel 3 — curvature    (0.17×): additive |curv_a − curv_b| penalty
//   channel 4 — direction    (0.09×): additive angular mismatch penalty
//   channel 5 — pressure     (0.15×): stylus/touch pressure profile mismatch
//
// Channel 5 is OPTIONAL: when neither point has pressure data (e.g. mouse),
// the contribution is 0 and the system degrades gracefully to 4-channel mode.
//
// A forgery that traces the correct overall bounding box but has a different
// internal curve structure CANNOT reduce channels 3–5 through warping.
// This breaks the calibration degeneracy that allowed shape forgeries to pass.

import type { StrokePoint } from "./types";

const SPATIAL_W   = 0.42;
const VELOCITY_W  = 0.17;
const CURVATURE_W = 0.17;
const DIRECTION_W = 0.09;
const PRESSURE_W  = 0.15;   // 0 contribution when pressure data absent

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
 * Multi-channel point distance: five additive independent channels.
 * Each channel can reveal a forgery independently — a forgery must match
 * ALL five to score well.
 *
 * Pressure channel activates only when BOTH points carry pressure data
 * (stylus or touch devices that report it). On mouse, contribution is 0.
 */
function multiChannelDist(
  a: StrokePoint, b: StrokePoint,
  curvA: number, curvB: number,
  dirA: number, dirB: number,
): number {
  const dx           = a.x - b.x;
  const dy           = a.y - b.y;
  const spatial      = Math.sqrt(dx * dx + dy * dy);
  const velDiff      = Math.abs((a.v ?? 0) - (b.v ?? 0));
  const curvDiff     = Math.abs(curvA - curvB);
  const dirDiff      = angleDiff(dirA, dirB);
  const pressureDiff = (a.pressure !== undefined && b.pressure !== undefined)
    ? Math.abs(a.pressure - b.pressure)
    : 0;

  return SPATIAL_W   * spatial
       + VELOCITY_W  * velDiff
       + CURVATURE_W * curvDiff
       + DIRECTION_W * dirDiff
       + PRESSURE_W  * pressureDiff;
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
 * Stroke-aware minimum score: compares corresponding strokes individually
 * and returns the WORST stroke score (minimum across all stroke pairs).
 *
 * This prevents a good "C" match from rescuing a bad "H vs T" mismatch
 * in a multi-letter signature — a forgery must match EVERY enrolled stroke.
 *
 * Decision rules:
 *   |refN − testN| > 1 → 20 (hard fail, completely different structure)
 *   |refN − testN| === 1 → min(per-stroke scores, 55) (soft penalty for extra/missing stroke)
 *   |refN − testN| === 0 → min(per-stroke scores) (pure quality gate)
 */
export function strokeMinScore(
  refStrokes: StrokePoint[][],
  testStrokes: StrokePoint[][],
  band: number = 0.15,
): number {
  const refN  = refStrokes.length;
  const testN = testStrokes.length;
  if (refN === 0 || testN === 0) return 100;

  if (Math.abs(refN - testN) > 1) return 20;

  const pairs = Math.min(refN, testN);
  let min = 100;
  for (let i = 0; i < pairs; i++) {
    const ref  = refStrokes[i];
    const test = testStrokes[i];
    if (ref.length === 0 || test.length === 0) continue;
    const raw   = curvatureDtw(ref, test, band);
    const score = curvatureDtwSimilarity(raw, ref.length, test.length);
    if (score < min) min = score;
  }

  if (refN !== testN) min = Math.min(min, 55);
  return min;
}

/**
 * Normalize multi-channel DTW distance to [0, 100] similarity score.
 *
 * Calibration constant K=115 (was 95) — tuned so that:
 *   - A genuine match (per-point ≈ 0.05) scores ~94
 *   - A medium forgery (per-point ≈ 0.30) scores ~65 (border)
 *   - A strong forgery (per-point ≈ 0.38) scores <56 (clear fail)
 *
 * Stricter than the old K=95 which allowed per-point ≈ 0.37 to pass
 * the 65-threshold — the "calibration K trap" from multi-channel DTW.
 */
export function curvatureDtwSimilarity(
  rawDistance: number,
  refLen: number,
  testLen: number,
): number {
  if (!isFinite(rawDistance) || rawDistance < 0) return 0;
  const avgLen = (refLen + testLen) / 2 || 1;
  const perPoint = rawDistance / avgLen;
  return Math.max(0, Math.min(100, Math.round((100 - perPoint * 115) * 10) / 10));
}
