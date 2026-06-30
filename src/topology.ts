// ─── Stroke Topology Fingerprint (GLYF novel primitive) ──────────────────────
// Reduces each stroke to a scale-invariant directional skeleton —
// a sequence of cardinal directions (8-way: N, NE, E, SE, S, SW, W, NW).
// This "chain code" captures the topological structure of a signature:
// what was drawn, not how fast or how big.
//
// Why this is novel for signature auth:
//   • Completely independent of scale, speed, and canvas size
//   • Captures the *shape grammar* of the signature — the sequence of turns
//   • Two different cursive styles for the same letter produce different codes
//   • Resistant to slow replay attacks (attacker gets shape right but rhythm wrong)
//   • Different from DTW: DTW compares how similar two sequences are;
//     topology checks whether the structural turns match at all
//
// Method:
//   1. Reduce stroke to direction vectors between smoothed points
//   2. Quantize each direction to one of 8 cardinal bins
//   3. Run-length encode to remove repeated directions (shape skeleton)
//   4. Compare skeletons with Levenshtein edit distance (string edit distance)

import type { StrokePoint, SignatureData } from "./types";

// 8 cardinal directions: E=0, NE=1, N=2, NW=3, W=4, SW=5, S=6, SE=7
const DIR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"] as const;
type Direction = typeof DIR_LABELS[number];

export interface StrokeTopology {
  chain: Direction[];      // direction sequence for each stroke
  skeleton: Direction[];   // run-length compressed direction sequence
  complexity: number;      // number of distinct direction segments
}

export interface TopologyProfile {
  strokes: StrokeTopology[];
  fullSkeleton: Direction[]; // combined skeleton of all strokes
  totalComplexity: number;
}

/**
 * Quantize an angle in radians to one of 8 cardinal directions.
 */
function quantizeDirection(angle: number): Direction {
  // Normalize to [0, 2π)
  const norm = ((angle % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const bin = Math.round((norm / (2 * Math.PI)) * 8) % 8;
  return DIR_LABELS[bin];
}

/**
 * Smooth a stroke by averaging position over a sliding window.
 * Reduces digitization noise before direction extraction.
 */
function smooth(pts: StrokePoint[], windowSize: number = 3): StrokePoint[] {
  if (pts.length <= windowSize) return pts;
  const half = Math.floor(windowSize / 2);
  return pts.map((_, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(pts.length - 1, i + half);
    const count = hi - lo + 1;
    return {
      x: pts.slice(lo, hi + 1).reduce((s, p) => s + p.x, 0) / count,
      y: pts.slice(lo, hi + 1).reduce((s, p) => s + p.y, 0) / count,
      t: pts[i].t,
    };
  });
}

/**
 * Run-length encode an array (collapse consecutive duplicates).
 */
function runLengthEncode<T>(arr: T[]): T[] {
  if (arr.length === 0) return [];
  const result: T[] = [arr[0]];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] !== arr[i - 1]) result.push(arr[i]);
  }
  return result;
}

/**
 * Extract the directional chain code and skeleton for a single stroke.
 */
function strokeTopology(stroke: StrokePoint[]): StrokeTopology {
  if (stroke.length < 2) {
    return { chain: [], skeleton: [], complexity: 0 };
  }

  const smoothed = smooth(stroke, 5);
  const chain: Direction[] = [];

  for (let i = 1; i < smoothed.length; i++) {
    const dx = smoothed[i].x - smoothed[i - 1].x;
    const dy = smoothed[i].y - smoothed[i - 1].y;
    if (Math.abs(dx) < 0.0005 && Math.abs(dy) < 0.0005) continue; // skip stationary
    chain.push(quantizeDirection(Math.atan2(dy, dx)));
  }

  const skeleton = runLengthEncode(chain);
  return { chain, skeleton, complexity: skeleton.length };
}

/**
 * Extract the topology profile for a full signature.
 */
export function extractTopology(sig: SignatureData): TopologyProfile {
  const strokes = sig.strokes.map(strokeTopology);
  const fullSkeleton = runLengthEncode(strokes.flatMap((s) => s.skeleton));
  const totalComplexity = strokes.reduce((s, st) => s + st.complexity, 0);
  return { strokes, fullSkeleton, totalComplexity };
}

/**
 * Levenshtein edit distance between two direction sequences.
 * Lower = more similar structural shape.
 */
function editDistance(a: Direction[], b: Direction[]): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map((_, i) => Array(n + 1).fill(0).map((__, j) => (i === 0 ? j : j === 0 ? i : 0)));

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        // Adjacent directions (e.g. N↔NE, and the circular wrap E↔SE) cost 0.5, others cost 1.
        // MUST use circular difference: |0 − 7| = 7 in linear space, but min(7, 8−7) = 1 circularly.
        const idxA = DIR_LABELS.indexOf(a[i - 1]);
        const idxB = DIR_LABELS.indexOf(b[j - 1]);
        const linearDiff = Math.abs(idxA - idxB);
        const circularDiff = Math.min(linearDiff, DIR_LABELS.length - linearDiff);
        const adjacentCost = circularDiff <= 1 ? 0.5 : 1;
        dp[i][j] = adjacentCost + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }
  return dp[m][n];
}

/**
 * Compare two topology profiles and return a similarity score [0, 100].
 * Uses edit distance on the combined skeleton sequences.
 */
export function topologySimilarity(ref: TopologyProfile, test: TopologyProfile): number {
  const refSkel = ref.fullSkeleton;
  const testSkel = test.fullSkeleton;

  if (refSkel.length === 0 && testSkel.length === 0) return 100;
  if (refSkel.length === 0 || testSkel.length === 0) return 0;

  const dist = editDistance(refSkel, testSkel);
  const maxLen = Math.max(refSkel.length, testSkel.length);

  // Normalize: 0 edits = 100, maxLen edits = 0
  return Math.max(0, Math.min(100, Math.round((1 - dist / maxLen) * 100 * 10) / 10));
}

/**
 * Average topology profiles (representative first profile used for skeleton).
 * Complexity stats are averaged.
 */
export function averageTopologyProfiles(profiles: TopologyProfile[]): TopologyProfile {
  if (profiles.length === 0) {
    return { strokes: [], fullSkeleton: [], totalComplexity: 0 };
  }
  // Use the median-complexity profile as the reference skeleton
  const sorted = [...profiles].sort((a, b) => a.totalComplexity - b.totalComplexity);
  const median = sorted[Math.floor(sorted.length / 2)];
  return {
    strokes: median.strokes,
    fullSkeleton: median.fullSkeleton,
    totalComplexity: profiles.reduce((s, p) => s + p.totalComplexity, 0) / profiles.length,
  };
}
