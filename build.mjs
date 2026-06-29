// Build script — generates ESM and CJS bundles from TypeScript source
import { build } from "esbuild";

const shared = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  sourcemap: true,
  target: "es2020",
  platform: "neutral",
  external: ["crypto"],
};

await build({ ...shared, format: "esm", outfile: "dist/index.js" });
await build({ ...shared, format: "cjs", outfile: "dist/index.cjs" });

console.log("Build complete: dist/index.js (ESM) + dist/index.cjs (CJS)");
