// ─── Feature extraction (GLYF v2) ────────────────────────────────────────────
// 14-dimension feature vector.
// v1 had 12 dimensions. v2 adds two novel biometric channels:
//   microtremorIndex      — high-frequency velocity variance (muscle tremor)
//   interStrokeRhythmRatio — mean inter-stroke pause as fraction of total duration
//
// These two channels capture signals that a spatial DTW has zero access to.
// A forger who draws slowly and deliberately has different microtremor than
// the authentic rapid cursive writer, and pauses between strokes at different
// ratios than the genuine signer — even if the drawn shape is correct.

import type { StrokePoint, SignatureData, FeatureVector } from "./types";

function pathLength(points: StrokePoint[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    total += Math.sqrt(dx * dx + dy * dy);
  }
  return total;
}

function boundingBox(points: StrokePoint[]) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { w: maxX - minX || 1, h: maxY - minY || 1, minX, minY };
}

function rawVelocities(points: StrokePoint[]): number[] {
  const vs: number[] = [];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const dt = Math.max(points[i].t - points[i - 1].t, 1);
    vs.push(Math.sqrt(dx * dx + dy * dy) / dt);
  }
  return vs;
}

function mean(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function variance(arr: number[]): number {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  return mean(arr.map((v) => (v - m) ** 2));
}

function curvatureEntropy(points: StrokePoint[]): number {
  if (points.length < 3) return 0;
  const angles: number[] = [];
  for (let i = 1; i < points.length - 1; i++) {
    const dx1 = points[i].x - points[i - 1].x;
    const dy1 = points[i].y - points[i - 1].y;
    const dx2 = points[i + 1].x - points[i].x;
    const dy2 = points[i + 1].y - points[i].y;
    angles.push(Math.abs(Math.atan2(dy2, dx2) - Math.atan2(dy1, dx1)));
  }
  const bins = new Array(8).fill(0);
  for (const a of angles) bins[Math.min(7, Math.floor((a / Math.PI) * 8))]++;
  const total = angles.length || 1;
  let entropy = 0;
  for (const c of bins) {
    if (c > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

function directionChangeRate(points: StrokePoint[]): number {
  if (points.length < 3) return 0;
  let changes = 0;
  let pdx = points[1].x - points[0].x;
  let pdy = points[1].y - points[0].y;
  for (let i = 2; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const dot = dx * pdx + dy * pdy;
    const mag = (Math.sqrt(pdx * pdx + pdy * pdy) || 1) * (Math.sqrt(dx * dx + dy * dy) || 1);
    if (dot / mag < -0.5) changes++;
    pdx = dx;
    pdy = dy;
  }
  return changes / points.length;
}

/**
 * Average velocity of the last 10% of a stroke — pen-lift deceleration pattern.
 */
function terminalVelocity(stroke: StrokePoint[]): number {
  if (stroke.length < 4) return 0;
  const tail = stroke.slice(Math.floor(stroke.length * 0.9));
  return mean(rawVelocities(tail));
}

/**
 * Mean absolute acceleration — how smoothly or jerkily the signer moves.
 */
function avgAcceleration(points: StrokePoint[]): number {
  const vs = rawVelocities(points);
  if (vs.length < 2) return 0;
  const accels: number[] = [];
  for (let i = 1; i < vs.length; i++) accels.push(Math.abs(vs[i] - vs[i - 1]));
  return mean(accels);
}

/**
 * GLYF v2 novel: Microtremor index — speed-invariant CV form.
 * Coefficient of variation of windowed velocity (W=5 points).
 * Formula: mean(window_variance) / (avgVelocity² + ε)
 *
 * Why CV-normalized (not raw variance):
 *   A genuine signer who signs 2× faster has 2× higher velocities and
 *   therefore 4× higher raw variance — which would wrongly flag them.
 *   Dividing by avgV² (coefficient of variation) removes the speed factor
 *   so the tremor PATTERN is compared, not the tremor magnitude.
 *   Forgers draw slowly/deliberately → abnormally smooth → CV near zero.
 *   Authentic rapid cursive → natural velocity jitter → CV clearly above zero.
 */
function microtremorIndex(points: StrokePoint[]): number {
  if (points.length < 10) return 0;
  const W = 5;
  const vs = rawVelocities(points);
  const avgV = mean(vs) || 0.001;
  const windowVars: number[] = [];
  for (let i = 0; i + W <= vs.length; i++) {
    windowVars.push(variance(vs.slice(i, i + W)));
  }
  // CV-normalized: speed-invariant tremor fingerprint
  return mean(windowVars) / (avgV * avgV + 0.0001);
}

/**
 * GLYF v2 novel: Inter-stroke rhythm ratio.
 * Mean inter-stroke pause duration normalized by total signing duration.
 * Each signer has a characteristic pause pattern — how long they lift the
 * pen between strokes. A forger concentrating on shape neglects this timing.
 */
function interStrokeRhythmRatio(sig: SignatureData): number {
  if (sig.strokes.length < 2 || sig.flatPoints.length < 2) return 0;
  const total =
    sig.flatPoints[sig.flatPoints.length - 1].t - sig.flatPoints[0].t || 1;
  let pauseSum = 0;
  let count = 0;
  for (let i = 0; i < sig.strokes.length - 1; i++) {
    const end = sig.strokes[i][sig.strokes[i].length - 1];
    const start = sig.strokes[i + 1][0];
    if (end && start) {
      pauseSum += Math.max(0, start.t - end.t);
      count++;
    }
  }
  return count > 0 ? Math.min(1, (pauseSum / count) / total) : 0;
}

export function extractFeatures(sig: SignatureData): FeatureVector {
  const pts = sig.flatPoints;
  if (pts.length < 3) {
    return {
      strokeCount: 0, aspectRatio: 1, totalPathLength: 0,
      avgVelocity: 0, velocityVariance: 0, peakVelocity: 0,
      avgAcceleration: 0, terminalVelocityProfile: 0,
      curvatureEntropy: 0, directionChangeRate: 0,
      durationMs: 0, strokeDurationVariance: 0,
      microtremorIndex: 0, interStrokeRhythmRatio: 0,
    };
  }

  const bbox = boundingBox(pts);
  const scale = Math.max(bbox.w, bbox.h) || 1;
  const vs = rawVelocities(pts);
  const duration = pts[pts.length - 1].t - pts[0].t || 1;
  const strokeDurations = sig.strokes.map((s) =>
    s.length > 1 ? s[s.length - 1].t - s[0].t : 0
  );

  return {
    strokeCount: sig.strokes.length,
    aspectRatio: bbox.w / bbox.h,
    totalPathLength: pathLength(pts) / scale,
    avgVelocity: mean(vs),
    velocityVariance: variance(vs),
    peakVelocity: Math.max(...vs, 0),
    avgAcceleration: avgAcceleration(pts),
    terminalVelocityProfile: mean(sig.strokes.map(terminalVelocity)),
    curvatureEntropy: curvatureEntropy(pts),
    directionChangeRate: directionChangeRate(pts),
    durationMs: duration,
    strokeDurationVariance: variance(strokeDurations),
    microtremorIndex: microtremorIndex(pts),
    interStrokeRhythmRatio: interStrokeRhythmRatio(sig),
  };
}

// ─── Feature channel weights (empirically benchmarked) ────────────────────────
//
// Benchmarked optimal configuration — 16.8 pt genuine/forgery separation:
//
//   strokeCount:           2.5  — hard gate on wrong pen-lift count
//   curvatureEntropy:      1.8  — sweet spot: catches hump-count forgeries without
//                                  penalising genuine large-noise writers (>1.8 drops
//                                  genuine-large-noise to ~73, too close to threshold)
//   microtremorIndex:      1.8  — CV-normalized tremor fingerprint; forgers draw slowly
//   interStrokeRhythmRatio: 2.0 — pause timing is the hardest channel to fake
//   directionChangeRate:   0    — EXCLUDED: too jitter-sensitive, hurts genuine users
//                                  more than it catches forgeries; net separation drops
//
// All other channels are secondary support — real discrimination load is on the
// four channels above plus the rhythm/angular/topology dedicated pipeline layers.
const WEIGHTS: Record<keyof FeatureVector, number> = {
  strokeCount:             2.5,
  aspectRatio:             1.5,
  totalPathLength:         1.2,
  avgVelocity:             0.7,
  velocityVariance:        0.6,
  peakVelocity:            0.5,
  avgAcceleration:         0.8,
  terminalVelocityProfile: 1.0,
  curvatureEntropy:        1.8,  // ↑ from 1.5 — benchmarked optimal
  directionChangeRate:     0,    // EXCLUDED — noise-sensitive, hurts genuines
  durationMs:              0.3,
  strokeDurationVariance:  0.7,
  microtremorIndex:        1.8,  // ↑ from 1.3 — benchmarked optimal
  interStrokeRhythmRatio:  2.0,  // ↑ from 1.5 — hardest channel to fake
};

export function featureSimilarity(ref: FeatureVector, test: FeatureVector): number {
  const keys = Object.keys(WEIGHTS) as (keyof FeatureVector)[];
  let weightedDist = 0;
  let totalWeight = 0;
  for (const key of keys) {
    const w = WEIGHTS[key];
    if (w === 0) continue; // skip excluded channels (e.g. directionChangeRate)
    const r = ref[key] as number;
    const t = test[key] as number;
    const scale = Math.max(Math.abs(r), Math.abs(t), 0.001);
    weightedDist += (Math.abs(r - t) / scale) * w;
    totalWeight += w;
  }
  return Math.max(0, Math.min(100, Math.round((100 - (weightedDist / totalWeight) * 80) * 10) / 10));
}

export function averageFeatureVectors(vectors: FeatureVector[]): FeatureVector {
  if (vectors.length === 0) throw new Error("No vectors to average");
  const keys = Object.keys(vectors[0]) as (keyof FeatureVector)[];
  const result = {} as FeatureVector;
  for (const key of keys) {
    result[key] = vectors.reduce((s, v) => s + (v[key] as number), 0) / vectors.length;
  }
  return result;
}

export function getFeatureWeights(): Record<keyof FeatureVector, number> {
  return { ...WEIGHTS };
}
