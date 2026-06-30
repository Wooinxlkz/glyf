# GLYF

> Signature biometric authentication library — curvature-weighted DTW, inter-stroke rhythm fingerprinting, angular momentum features, stroke topology fingerprinting, adversarial forgery simulation, and explainable rejection.

GLYF is a TypeScript library for building signature-based authentication into web and Node.js applications. It runs entirely client-side with no server round-trips for verification, and produces non-invertible templates safe to store in a database.

---

## What makes it different

| Feature | GLYF | Baseline DTW libs |
|---|---|---|
| DTW variant | **Curvature-weighted** (corners matter more than straight segments) | Standard, equal-weight |
| Sakoe-Chiba band | **Per-user adaptive** (from enrollment velocity variance) | Fixed 10–15% |
| Score channels | **6 channels** (shape, features, rhythm, autocorr, angular, topology) | 1–2 channels |
| Rhythm channel | **Inter-stroke pause histogram + autocorrelation** | Not present |
| Angular channel | **Angular momentum per stroke** (rotational pen dynamics) | Not present |
| Topology channel | **Scale-invariant stroke skeleton** (Levenshtein edit distance) | Not present |
| Security testing | **Adversarial forgery simulation — 7 attack types, fully automatic** | Manual user spoofing |
| Rejection feedback | **Explainable per-channel failure reasons in plain language** | Pass/fail score only |
| Template storage | **Non-invertible SHA-256 hash of quantized features** | Raw feature vectors |

---

## Install

```bash
npm install glyf
# or
pnpm add glyf
```

---

## Quick start

```ts
import { enroll, verify } from "glyf";
import type { SignatureData } from "glyf";

// 1. Enroll — collect 3+ signature samples
const template = enroll([sample1, sample2, sample3]);

// 2. Verify — check a new signature
const result = verify(template, newSig);

console.log(result.explained.passed);       // true / false
console.log(result.explained.label);        // "Verified (adaptive)" or "Rejected — ..."
console.log(result.combinedScore);          // 0-100
console.log(result.shapeScore);             // curvature-weighted DTW channel
console.log(result.angularScore);           // angular momentum channel
console.log(result.topologyScore);          // stroke skeleton channel
console.log(result.matchStrict);            // score >= 82
console.log(result.matchAdaptive);          // score >= user's personal threshold
```

### Building a `SignatureData` from `signature_pad`

```ts
import SignaturePad from "signature_pad";
import type { SignatureData } from "glyf";

function capture(pad: SignaturePad): SignatureData {
  const strokes = pad.toData().map((group) =>
    group.points.map((p) => ({ x: p.x, y: p.y, t: p.time, pressure: p.pressure }))
  );
  return {
    strokes,
    flatPoints: strokes.flat(),
    capturedAt: Date.now(),
    canvasWidth: pad.canvas.width,
    canvasHeight: pad.canvas.height,
    devicePixelRatio: window.devicePixelRatio,
  };
}
```

---

## Novel algorithms

### 1. Curvature-Weighted DTW
Standard DTW weights every point equally. GLYF computes the local curvature at each point (how sharply the path bends) and uses it as a per-point weight in the distance metric. Corners and loops — the parts that make a signature unique — contribute more to the match score. Straight filler segments contribute less.

```ts
import { curvatureDtw, curvatureDtwSimilarity } from "glyf";

const rawDist = curvatureDtw(refSequence, testSequence, bandFraction);
const score = curvatureDtwSimilarity(rawDist, refSequence.length, testSequence.length);
```

### 2. Per-User Adaptive Sakoe-Chiba Band
The band (how much DTW is allowed to warp) is computed from the signer's own velocity variance during enrollment — not a fixed constant.

- Consistent signers → narrow band (≈8%) → tighter gate, harder to fool
- Naturally variable signers → wider band (≈28%) → prevents unfair rejection

```ts
import { computeAdaptiveBand, preprocess } from "glyf";

const processed = samples.map((s) => preprocess(s.flatPoints, s));
const band = computeAdaptiveBand(processed); // unique to this user
```

### 3. Inter-Stroke Rhythm Fingerprinting
The pause between strokes is treated as a separate biometric channel. Pause durations are normalized by total signing time, turned into an 8-bin histogram, and compared with Bhattacharyya coefficient.

```ts
import { extractRhythm, rhythmSimilarity } from "glyf";

const refRhythm = extractRhythm(referenceSig);
const testRhythm = extractRhythm(testSig);
const score = rhythmSimilarity(refRhythm, testRhythm); // 0-100
```

### 4. Rhythmic Autocorrelation
Goes further than a histogram — computes the autocorrelation of the pause sequence to detect periodic rhythm patterns (e.g. "always pauses longer before the 3rd stroke"). Compared via cosine similarity.

```ts
import { extractAutocorr, autocorrSimilarity } from "glyf";

const refAC = extractAutocorr(refRhythm);
const testAC = extractAutocorr(testRhythm);
const score = autocorrSimilarity(refAC, testAC); // 0-100
```

### 5. Angular Momentum Features
Treats the pen tip as a particle and computes angular momentum around each stroke's centroid. Captures whether letters are drawn clockwise or counterclockwise, and how energetically the signer curves.

```ts
import { extractAngular, angularSimilarity } from "glyf";

const refAngular = extractAngular(referenceSig);
const testAngular = extractAngular(testSig);
const score = angularSimilarity(refAngular, testAngular); // 0-100
```

### 6. Stroke Topology Fingerprint
Reduces each stroke to a scale-invariant directional skeleton — a chain code of 8 cardinal directions (N, NE, E, SE, S, SW, W, NW). Compared using Levenshtein edit distance with adjacent-direction cost of 0.5. Completely independent of scale, speed, and canvas size.

```ts
import { extractTopology, topologySimilarity } from "glyf";

const refTopo = extractTopology(referenceSig);
const testTopo = extractTopology(testSig);
const score = topologySimilarity(refTopo, testTopo); // 0-100
```

### 7. Adversarial Forgery Simulation
Auto-generates 12+ synthetic forgeries from the enrolled template and measures the simulated False Acceptance Rate — no human spoofing needed.

Attack types: spatial jitter (3 intensities), speed attacks (4 factors), stroke omission, shape stretch, replay from enrollment, velocity smoothing.

```ts
import { runAdversarialSimulation } from "glyf";

const report = runAdversarialSimulation(
  refProcessed, rawSamples, rawStrokes,
  template.adaptiveThreshold, template.adaptiveBandFraction
);

console.log(report.simulatedFAR); // e.g. 0.083
console.log(report.verdict);      // "SECURE" | "MARGINAL" | "VULNERABLE"
```

---

## Channel weights

| Channel | Weight | Captures |
|---|---|---|
| Curvature-weighted DTW | 38% | Path shape with emphasis on distinctive curves |
| Feature vector (12-dim) | 25% | Speed, curvature, stroke count, acceleration |
| Rhythm histogram | 12% | Inter-stroke pause distribution |
| Rhythm autocorrelation | 5% | Periodic pause patterns |
| Angular momentum | 12% | Rotational pen dynamics per stroke |
| Stroke topology | 8% | Scale-invariant direction skeleton |

---

## Threshold tiers

| Tier | Score | Use case |
|---|---|---|
| Strict | ≥ 82 | High-security transactions |
| Adaptive | ≥ personal threshold | Default, recommended |
| Loose | ≥ 50 | Accessibility |

---

## Template storage

```ts
import { hashTemplate, safeStorageTemplate } from "glyf";

// Async SHA-256 (browser SubtleCrypto or Node crypto)
const hash = await hashTemplate(template.avgFeatures, "your-app-salt");

// Safe fields to store in DB
const safe = safeStorageTemplate(template.avgFeatures);
```

Never store raw coordinates or timing data. Only store the hash and safe fields.

---

## Build

```bash
npm install
npm run build
# Output: dist/index.js (ESM) + dist/index.cjs (CJS)
```

---

## License

MIT © 2026 nulltrace - Karim
