// ─── Device-agnostic normalization ───────────────────────────────────────────
// Maps raw captured points to a canonical [0,1]×[0,1] space regardless of
// canvas size, DPI, or input device (mouse, touch, stylus).

import type { StrokePoint, SignatureData } from "./types";

const SAMPLE_SIZE = 128;

function boundingBox(points: StrokePoint[]) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { minX, maxX, minY, maxY, w: maxX - minX || 1, h: maxY - minY || 1 };
}

/**
 * Normalize coordinates into [0,1]×[0,1] preserving aspect ratio.
 * If canvas dimensions are provided, normalizes relative to the canvas first
 * so a signature in the top-left of a large canvas equals one centered
 * on a small canvas (both still reflect the signer's pattern, not canvas position).
 */
export function normalizeCoords(
  points: StrokePoint[],
  sig?: Pick<SignatureData, "canvasWidth" | "canvasHeight" | "devicePixelRatio">
): StrokePoint[] {
  if (points.length === 0) return [];

  let pts = points;

  // If canvas info is available, map out of device-pixel space first
  if (sig?.canvasWidth && sig?.canvasHeight) {
    const dpr = sig.devicePixelRatio ?? 1;
    const cw = sig.canvasWidth / dpr;
    const ch = sig.canvasHeight / dpr;
    pts = points.map((p) => ({ ...p, x: p.x / cw, y: p.y / ch }));
  }

  const bbox = boundingBox(pts);
  const scale = Math.max(bbox.w, bbox.h) || 1;

  return pts.map((p) => ({
    ...p,
    x: (p.x - bbox.minX) / scale,
    y: (p.y - bbox.minY) / scale,
  }));
}

/**
 * Normalize timestamps to [0,1] space across the full signing duration.
 * Lets rhythm comparisons work across fast vs slow signers.
 */
export function normalizeTime(points: StrokePoint[]): StrokePoint[] {
  if (points.length === 0) return [];
  const t0 = points[0].t;
  const tMax = points[points.length - 1].t - t0 || 1;
  return points.map((p) => ({ ...p, t: (p.t - t0) / tMax }));
}

/**
 * Compute normalized velocity for each point in [0,1] using spatial distance
 * over raw (non-normalized) time. Must be called AFTER normalizeCoords.
 */
export function addVelocity(points: StrokePoint[]): StrokePoint[] {
  if (points.length === 0) return [];
  const speeds: number[] = [0];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const dt = Math.max(points[i].t - points[i - 1].t, 1);
    speeds.push(Math.sqrt(dx * dx + dy * dy) / dt);
  }
  const maxSpeed = Math.max(...speeds, 0.0001);
  return points.map((p, i) => ({ ...p, v: speeds[i] / maxSpeed }));
}

/**
 * Downsample to a fixed point count using linear interpolation index mapping.
 * Preserves the relative distribution of points along the path.
 */
export function downsample(points: StrokePoint[], target: number = SAMPLE_SIZE): StrokePoint[] {
  if (points.length <= target) return points;
  const result: StrokePoint[] = [];
  const step = (points.length - 1) / (target - 1);
  for (let i = 0; i < target; i++) {
    result.push(points[Math.round(i * step)]);
  }
  return result;
}

/**
 * Full preprocessing pipeline used before DTW comparison.
 * coord-normalize → add velocity → downsample
 */
export function preprocess(
  flatPoints: StrokePoint[],
  sig?: Pick<SignatureData, "canvasWidth" | "canvasHeight" | "devicePixelRatio">
): StrokePoint[] {
  return downsample(addVelocity(normalizeCoords(flatPoints, sig)), SAMPLE_SIZE);
}
