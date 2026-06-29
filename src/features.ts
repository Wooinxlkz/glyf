// ─── Feature extraction ───────────────────────────────────────────────────────
// Extended feature vector vs baseline: adds acceleration profile,
// terminal velocity per stroke, and stroke duration variance.

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
 * Average velocity of the last 10% of points in a stroke —
 * captures the "pen-lift" deceleration pattern unique to each signer.
 */
function terminalVelocity(stroke: StrokePoint[]): number {
  if (stroke.length < 4) return 0;
  const tail = stroke.slice(Math.floor(stroke.length * 0.9));
  const vs = rawVelocities(tail);
  return mean(vs);
}

/**
 * Mean absolute acceleration (rate of velocity change).
 * Captures how smoothly or jerkily the signer moves.
 */
function avgAcceleration(points: StrokePoint[]): number {
  const vs = rawVelocities(points);
  if (vs.length < 2) return 0;
  const accels: number[] = [];
  for (let i = 1; i < vs.length; i++) {
    accels.push(Math.abs(vs[i] - vs[i - 1]));
  }
  return mean(accels);
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
    };
  }

  const bbox = boundingBox(pts);
  const scale = Math.max(bbox.w, bbox.h) || 1;
  const vs = rawVelocities(pts);
  const duration = pts[pts.length - 1].t - pts[0].t || 1;

  // Per-stroke durations variance
  const strokeDurations = sig.strokes.map((s) =>
    s.length > 1 ? s[s.length - 1].t - s[0].t : 0
  );
  const tvProfile = mean(sig.strokes.map(terminalVelocity));

  return {
    strokeCount: sig.strokes.length,
    aspectRatio: bbox.w / bbox.h,
    totalPathLength: pathLength(pts) / scale,
    avgVelocity: mean(vs),
    velocityVariance: variance(vs),
    peakVelocity: Math.max(...vs, 0),
    avgAcceleration: avgAcceleration(pts),
    terminalVelocityProfile: tvProfile,
    curvatureEntropy: curvatureEntropy(pts),
    directionChangeRate: directionChangeRate(pts),
    durationMs: duration,
    strokeDurationVariance: variance(strokeDurations),
  };
}

// Channel weights — more channels means lower individual weight
const WEIGHTS: Record<keyof FeatureVector, number> = {
  strokeCount: 2.0,
  aspectRatio: 1.5,
  totalPathLength: 1.2,
  avgVelocity: 0.7,
  velocityVariance: 0.6,
  peakVelocity: 0.5,
  avgAcceleration: 0.8,       // NEW
  terminalVelocityProfile: 1.0, // NEW
  curvatureEntropy: 1.5,
  directionChangeRate: 1.0,
  durationMs: 0.3,
  strokeDurationVariance: 0.7, // NEW
};

export function featureSimilarity(ref: FeatureVector, test: FeatureVector): number {
  const keys = Object.keys(WEIGHTS) as (keyof FeatureVector)[];
  let weightedDist = 0;
  let totalWeight = 0;
  for (const key of keys) {
    const r = ref[key] as number;
    const t = test[key] as number;
    const w = WEIGHTS[key];
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
