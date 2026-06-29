// ─── Non-invertible template hashing ─────────────────────────────────────────
// Raw biometric coordinates must NEVER be stored in a database.
// This module produces a one-way hash of the quantized feature vector
// that can be stored safely — it cannot be reversed to reconstruct
// the original signature coordinates.
//
// Approach: quantize each feature to discrete bins → concatenate → SHA-256.
// A different feature value will produce a completely different hash
// (no gradient — not reversible by hill-climbing either).

import type { FeatureVector } from "./types";

// Quantization step sizes per feature.
// Chosen to preserve enough granularity that different signers produce
// different hashes while being tolerant of small natural variation.
const QUANT: Record<keyof FeatureVector, number> = {
  strokeCount: 1,
  aspectRatio: 0.05,
  totalPathLength: 0.05,
  avgVelocity: 0.01,
  velocityVariance: 0.001,
  peakVelocity: 0.02,
  avgAcceleration: 0.005,
  terminalVelocityProfile: 0.01,
  curvatureEntropy: 0.1,
  directionChangeRate: 0.01,
  durationMs: 100,
  strokeDurationVariance: 50,
};

function quantize(value: number, step: number): number {
  return Math.round(value / step) * step;
}

function featureString(features: FeatureVector): string {
  const keys = Object.keys(QUANT) as (keyof FeatureVector)[];
  return keys
    .map((k) => `${k}:${quantize(features[k] as number, QUANT[k]).toFixed(6)}`)
    .join("|");
}

/**
 * Compute a SHA-256 hash of the quantized feature vector.
 * Works in both browser (SubtleCrypto) and Node.js (crypto module).
 * Returns a hex string.
 */
export async function hashTemplate(
  features: FeatureVector,
  salt: string = "penprint-v1"
): Promise<string> {
  const payload = `${salt}::${featureString(features)}`;
  const encoded = new TextEncoder().encode(payload);

  // Browser environment
  if (typeof globalThis.crypto?.subtle !== "undefined") {
    const hashBuffer = await globalThis.crypto.subtle.digest("SHA-256", encoded);
    return Array.from(new Uint8Array(hashBuffer))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  // Node.js environment
  const { createHash } = await import("crypto");
  return createHash("sha256").update(payload).digest("hex");
}

/**
 * Synchronous hash using a simple FNV-1a 64-bit approximation.
 * Use this when async is not available. Less collision-resistant
 * than SHA-256 but still non-invertible for practical purposes.
 */
export function hashTemplateSync(
  features: FeatureVector,
  salt: string = "penprint-v1"
): string {
  const payload = `${salt}::${featureString(features)}`;
  let h1 = 0x811c9dc5;
  let h2 = 0xc3a2d88b;
  for (let i = 0; i < payload.length; i++) {
    const c = payload.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193);
    h2 = Math.imul(h2 ^ c, 0x01000193) ^ (h1 >>> 16);
  }
  return `${(h1 >>> 0).toString(16).padStart(8, "0")}${(h2 >>> 0).toString(16).padStart(8, "0")}`;
}

/**
 * Returns only the safe-to-store fields from a feature vector.
 * Omits raw timing fields that could reveal the device or context.
 */
export function safeStorageTemplate(features: FeatureVector): Partial<FeatureVector> {
  return {
    strokeCount: Math.round(features.strokeCount),
    aspectRatio: Math.round(features.aspectRatio * 100) / 100,
    totalPathLength: Math.round(features.totalPathLength * 1000) / 1000,
    curvatureEntropy: Math.round(features.curvatureEntropy * 1000) / 1000,
    directionChangeRate: Math.round(features.directionChangeRate * 1000) / 1000,
    terminalVelocityProfile: Math.round(features.terminalVelocityProfile * 1000) / 1000,
  };
}
