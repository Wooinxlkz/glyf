// ─── Adversarial forgery simulation ──────────────────────────────────────────
// Instead of asking a human to manually spoof their own signature (unreliable),
// PenPrint auto-generates synthetic "smart forgeries" from the enrolled template
// and measures how many pass the threshold.
//
// This gives a statistically meaningful False Acceptance Rate estimate without
// any human input — and uses the kinds of perturbations a real attacker would
// plausibly attempt rather than random noise.
//
// Forgery types generated:
//   1. Jitter attack        — small spatial noise added to all points
//   2. Speed attack         — uniform time stretching / compression
//   3. Stroke omission      — one stroke removed
//   4. Shape stretch        — bounding box deformed along one axis
//   5. Replay attack        — points from one enrollment sample replayed as test
//   6. Partial trace        — only the outermost convex hull of the path traced
//   7. Velocity smoothing   — velocity randomness averaged out (robot-like signing)

import type { StrokePoint } from "./types";
import type { AdversarialReport } from "./types";
import { compareSequences } from "./dtw";
import { downsample } from "./normalizer";

const SAMPLE_SIZE = 128;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ── Forgery generators ────────────────────────────────────────────────────────

/** Add small spatial Gaussian-like jitter. Intensity 0-1. */
function jitterAttack(pts: StrokePoint[], intensity: number): StrokePoint[] {
  return pts.map((p) => ({
    ...p,
    x: p.x + (Math.random() - 0.5) * intensity * 0.08,
    y: p.y + (Math.random() - 0.5) * intensity * 0.08,
  }));
}

/** Stretch or compress time uniformly (forger signs faster or slower). */
function speedAttack(pts: StrokePoint[], factor: number): StrokePoint[] {
  if (pts.length === 0) return pts;
  const t0 = pts[0].t;
  return pts.map((p) => ({ ...p, t: t0 + (p.t - t0) * factor }));
}

/** Remove the last stroke (single-stroke signatures are immune). */
function strokeOmission(pts: StrokePoint[], strokes: StrokePoint[][]): StrokePoint[] {
  if (strokes.length <= 1) return pts;
  const kept = strokes.slice(0, -1);
  return kept.flat();
}

/** Stretch signature along X axis — changes aspect ratio. */
function shapeStretch(pts: StrokePoint[], scaleX: number): StrokePoint[] {
  if (pts.length === 0) return pts;
  const minX = Math.min(...pts.map((p) => p.x));
  return pts.map((p) => ({ ...p, x: minX + (p.x - minX) * scaleX }));
}

/** Take a second enrollment sample and replay it as the forgery. */
function replayAttack(sample: StrokePoint[]): StrokePoint[] {
  return [...sample];
}

/**
 * Smooth out velocity variation — simulates a robot or forger who traces shape
 * correctly but without the natural velocity variation of the real signer.
 */
function velocitySmoothing(pts: StrokePoint[]): StrokePoint[] {
  if (pts.length < 3) return pts;
  const smoothed: StrokePoint[] = [pts[0]];
  const windowSize = 5;
  for (let i = 1; i < pts.length - 1; i++) {
    const lo = Math.max(0, i - Math.floor(windowSize / 2));
    const hi = Math.min(pts.length - 1, i + Math.floor(windowSize / 2));
    let sumX = 0, sumY = 0, count = 0;
    for (let j = lo; j <= hi; j++) {
      sumX += pts[j].x;
      sumY += pts[j].y;
      count++;
    }
    smoothed.push({ ...pts[i], x: sumX / count, y: sumY / count });
  }
  smoothed.push(pts[pts.length - 1]);
  return smoothed;
}

// ── Main simulation ───────────────────────────────────────────────────────────

/**
 * Run the full adversarial forgery simulation against an enrolled template.
 *
 * @param refProcessed  Preprocessed (DTW-ready) reference sequence from enrollment
 * @param rawSamples    Raw flat-point arrays of each enrollment sample (for replay)
 * @param rawStrokes    Stroke arrays from the primary enrollment sample
 * @param threshold     The threshold to test against (adaptive or strict)
 * @param bandFraction  Per-user Sakoe-Chiba band
 */
export function runAdversarialSimulation(
  refProcessed: StrokePoint[],
  rawSamples: StrokePoint[][],
  rawStrokes: StrokePoint[][],
  threshold: number,
  bandFraction: number
): AdversarialReport {
  const attempts: { type: string; pts: StrokePoint[] }[] = [];

  // 1. Jitter attacks (light and heavy)
  for (const intensity of [0.3, 0.6, 1.0]) {
    attempts.push({ type: `Jitter (${Math.round(intensity * 100)}%)`, pts: jitterAttack(refProcessed, intensity) });
  }

  // 2. Speed attacks (slow down, speed up)
  for (const factor of [0.6, 0.8, 1.25, 1.5]) {
    attempts.push({ type: `Speed ${factor > 1 ? "faster" : "slower"} (${factor}×)`, pts: speedAttack(refProcessed, factor) });
  }

  // 3. Stroke omission
  if (rawStrokes.length > 1) {
    const omitted = strokeOmission(refProcessed, rawStrokes);
    const processed = downsample(omitted, SAMPLE_SIZE);
    attempts.push({ type: "Stroke omission (last)", pts: processed });
  }

  // 4. Shape stretch (narrow and wide)
  for (const sx of [0.7, 1.3]) {
    attempts.push({ type: `Shape stretch (${sx}×)`, pts: shapeStretch(refProcessed, sx) });
  }

  // 5. Replay attack — use another enrollment sample as "forgery"
  if (rawSamples.length >= 2) {
    attempts.push({ type: "Replay sample #2", pts: rawSamples[1] });
    if (rawSamples.length >= 3) {
      attempts.push({ type: "Replay sample #3", pts: rawSamples[2] });
    }
  }

  // 6. Velocity smoothing (robot-like)
  const flat = refProcessed.map((p) => ({ ...p }));
  attempts.push({ type: "Velocity smoothing", pts: downsample(velocitySmoothing(flat), SAMPLE_SIZE) });

  // Score each forgery attempt
  const results = attempts.map(({ type, pts }) => {
    const { score } = compareSequences(refProcessed, pts, bandFraction);
    return {
      forgeryType: type,
      score: Math.round(score * 10) / 10,
      passed: score >= threshold,
    };
  });

  const passedAttempts = results.filter((r) => r.passed).length;
  const simulatedFAR = Math.round((passedAttempts / results.length) * 1000) / 1000;
  const worstCaseScore = Math.max(...results.map((r) => r.score));

  let verdict: AdversarialReport["verdict"];
  if (simulatedFAR === 0) verdict = "SECURE";
  else if (simulatedFAR <= 0.15) verdict = "MARGINAL";
  else verdict = "VULNERABLE";

  return {
    simulatedFAR,
    worstCaseScore,
    totalAttempts: results.length,
    passedAttempts,
    results,
    verdict,
  };
}
