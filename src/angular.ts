// ─── Angular Momentum Features (GLYF novel primitive) ────────────────────────
// Treats the pen tip as a particle moving through 2D space.
// For each stroke, computes the angular momentum of the pen about the
// stroke's centroid — the rotational "spin" of the writing motion.
//
// Why this is novel for signature auth:
//   • Captures whether letters are drawn clockwise or counterclockwise
//   • Magnitude captures how energetically the signer curves
//   • A forger tracing the correct shape won't replicate the angular dynamics
//     unless they also match the exact speed at each curve — very hard to fake
//
// Angular momentum at point i:
//   L_i = r_i × v_i  (cross product of displacement from centroid × velocity)
//       = (x_i - cx)(vy_i) - (y_i - cy)(vx_i)
// where cx, cy = stroke centroid, vx/vy = velocity components.

import type { StrokePoint, SignatureData } from "./types";

export interface AngularProfile {
  perStroke: number[];     // net angular momentum per stroke (signed)
  totalAngular: number;    // sum of absolute angular momenta
  angularVariance: number; // variance across strokes
  dominantDirection: "CW" | "CCW" | "MIXED"; // clockwise, counter-clockwise
}

function centroid(pts: StrokePoint[]): { cx: number; cy: number } {
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  return { cx, cy };
}

/**
 * Compute net angular momentum of a single stroke about its centroid.
 * Returns a signed value: positive = CCW, negative = CW.
 */
function strokeAngularMomentum(stroke: StrokePoint[]): number {
  if (stroke.length < 3) return 0;
  const { cx, cy } = centroid(stroke);
  let L = 0;

  for (let i = 1; i < stroke.length; i++) {
    const dt = Math.max(stroke[i].t - stroke[i - 1].t, 1);
    const vx = (stroke[i].x - stroke[i - 1].x) / dt;
    const vy = (stroke[i].y - stroke[i - 1].y) / dt;
    // Displacement from centroid (use mid-point for stability)
    const rx = (stroke[i].x + stroke[i - 1].x) / 2 - cx;
    const ry = (stroke[i].y + stroke[i - 1].y) / 2 - cy;
    // Cross product: r × v
    L += rx * vy - ry * vx;
  }
  return L / stroke.length; // normalize by stroke length
}

/**
 * Extract angular momentum profile from all strokes in a signature.
 */
export function extractAngular(sig: SignatureData): AngularProfile {
  const strokes = sig.strokes;
  if (strokes.length === 0) {
    return { perStroke: [], totalAngular: 0, angularVariance: 0, dominantDirection: "MIXED" };
  }

  const perStroke = strokes.map(strokeAngularMomentum);
  const absValues = perStroke.map(Math.abs);
  const totalAngular = absValues.reduce((a, b) => a + b, 0);

  const mean = totalAngular / perStroke.length;
  const angularVariance =
    absValues.reduce((a, b) => a + (b - mean) ** 2, 0) / perStroke.length;

  const sumSigned = perStroke.reduce((a, b) => a + b, 0);
  const dominantDirection: AngularProfile["dominantDirection"] =
    Math.abs(sumSigned) < totalAngular * 0.2
      ? "MIXED"
      : sumSigned > 0
        ? "CCW"
        : "CW";

  return { perStroke, totalAngular, angularVariance, dominantDirection };
}

/**
 * Compare two angular profiles and return a similarity score [0, 100].
 *
 * Three components:
 *   1. Per-stroke angular momentum profile similarity (cosine similarity)
 *   2. Total angular energy similarity
 *   3. Direction match bonus
 */
export function angularSimilarity(ref: AngularProfile, test: AngularProfile): number {
  // If either has no strokes, skip angular channel
  if (ref.perStroke.length === 0 || test.perStroke.length === 0) return 100;

  // 1. Per-stroke cosine similarity
  const n = Math.min(ref.perStroke.length, test.perStroke.length);
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < n; i++) {
    dot += ref.perStroke[i] * test.perStroke[i];
    magA += ref.perStroke[i] ** 2;
    magB += test.perStroke[i] ** 2;
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  const cosine = denom > 0 ? dot / denom : 0;
  const profileScore = ((cosine + 1) / 2) * 100;

  // 2. Total angular energy similarity
  const refE = ref.totalAngular;
  const testE = test.totalAngular;
  const energyScale = Math.max(refE, testE, 0.001);
  const energyScore = Math.max(0, 100 - (Math.abs(refE - testE) / energyScale) * 100);

  // 3. Direction match
  const directionMatch = ref.dominantDirection === test.dominantDirection ? 100 : 40;

  return Math.round((profileScore * 0.5 + energyScore * 0.3 + directionMatch * 0.2) * 10) / 10;
}

/**
 * Average multiple angular profiles into a reference.
 */
export function averageAngularProfiles(profiles: AngularProfile[]): AngularProfile {
  const valid = profiles.filter((p) => p.perStroke.length > 0);
  if (valid.length === 0) {
    return { perStroke: [], totalAngular: 0, angularVariance: 0, dominantDirection: "MIXED" };
  }

  const minLen = Math.min(...valid.map((p) => p.perStroke.length));
  const perStroke = Array.from({ length: minLen }, (_, i) =>
    valid.reduce((s, p) => s + p.perStroke[i], 0) / valid.length
  );
  const totalAngular = valid.reduce((s, p) => s + p.totalAngular, 0) / valid.length;
  const angularVariance = valid.reduce((s, p) => s + p.angularVariance, 0) / valid.length;

  const directions = valid.map((p) => p.dominantDirection);
  const dominantDirection = directions.every((d) => d === directions[0])
    ? directions[0]
    : "MIXED";

  return { perStroke, totalAngular, angularVariance, dominantDirection };
}
