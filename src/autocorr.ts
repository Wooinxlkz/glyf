// ─── Rhythmic Autocorrelation (GLYF novel primitive) ─────────────────────────
// Standard rhythm fingerprinting uses a histogram of inter-stroke pause
// durations. GLYF goes further: it computes the autocorrelation of the
// pause sequence, capturing whether a signer has *periodic* rhythm patterns
// (e.g. "my 3rd pause is always ≈ 2× my 1st pause").
//
// This is novel because:
//   • It detects periodic structure that a histogram cannot see
//   • It's invariant to absolute signing speed (ratios matter, not raw ms)
//   • A forger who gets the shape right will almost never replicate
//     the autocorrelation profile
//
// The output is a fixed-length autocorrelation vector at lags 1..MAX_LAG,
// which can be compared with cosine similarity.

import type { RhythmProfile } from "./types";

const MAX_LAG = 5; // lags beyond 5 strokes are not meaningful for short sigs

/**
 * Compute the normalized autocorrelation of a 1D sequence at lags 1..MAX_LAG.
 * Returns a vector of length MAX_LAG with values in [-1, 1].
 * If the sequence is too short, returns zeros.
 */
function autocorrelation(seq: number[], maxLag: number): number[] {
  const n = seq.length;
  const result: number[] = new Array(maxLag).fill(0);
  if (n < 2) return result;

  const mean = seq.reduce((a, b) => a + b, 0) / n;
  const variance = seq.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  if (variance === 0) return result;

  for (let lag = 1; lag <= maxLag && lag < n; lag++) {
    let cov = 0;
    for (let i = 0; i < n - lag; i++) {
      cov += (seq[i] - mean) * (seq[i + lag] - mean);
    }
    result[lag - 1] = cov / ((n - lag) * variance);
  }
  return result;
}

export interface RhythmAutocorr {
  vector: number[];       // autocorrelation at lags 1..MAX_LAG
  dominantLag: number;    // lag with highest positive autocorrelation
  periodicity: number;    // 0-1, how periodic the rhythm is
}

/**
 * Extract the rhythmic autocorrelation profile from a rhythm profile.
 * Input is a RhythmProfile already extracted by rhythm.ts.
 */
export function extractAutocorr(rhythm: RhythmProfile): RhythmAutocorr {
  const seq = rhythm.pauseRatios;
  if (seq.length < 2) {
    return { vector: new Array(MAX_LAG).fill(0), dominantLag: 0, periodicity: 0 };
  }

  const vector = autocorrelation(seq, MAX_LAG);
  const maxVal = Math.max(...vector);
  const dominantLag = maxVal > 0.1 ? vector.indexOf(maxVal) + 1 : 0;
  const periodicity = Math.max(0, maxVal); // 0 = no periodicity, 1 = perfectly periodic

  return { vector, dominantLag, periodicity };
}

/**
 * Compare two rhythmic autocorrelation profiles using cosine similarity.
 * Returns a similarity score in [0, 100].
 *
 * Cosine similarity is ideal here: it measures whether the two signers
 * share the same rhythm *shape* regardless of magnitude.
 */
export function autocorrSimilarity(a: RhythmAutocorr, b: RhythmAutocorr): number {
  const va = a.vector;
  const vb = b.vector;
  const n = Math.min(va.length, vb.length);
  if (n === 0) return 100; // single-stroke — no rhythm to compare

  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < n; i++) {
    dot += va[i] * vb[i];
    magA += va[i] * va[i];
    magB += vb[i] * vb[i];
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 100; // both zero vectors = both non-periodic = match

  // cosine ∈ [-1, 1] → score ∈ [0, 100]
  const cosine = dot / denom;
  return Math.round(((cosine + 1) / 2) * 100 * 10) / 10;
}
