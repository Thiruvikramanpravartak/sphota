// The browser refinement (src/refine.ts) must give exactly the Python library's groups on the
// shared fixture (tests/fixtures/refine_parity.json, written by scripts/make_refine_fixture.py).
//   node tests/refine-parity.mjs        (CI: atlas job)
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, renameSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const out = mkdtempSync(join(tmpdir(), "refine-"));
execFileSync("npx", ["tsc", "src/refine.ts", "--ignoreConfig", "--outDir", out, "--module", "es2022",
  "--target", "es2022", "--lib", "es2022,dom", "--skipLibCheck"], { stdio: "inherit" });
renameSync(join(out, "refine.js"), join(out, "refine.mjs"));
const { refineFile } = await import(pathToFileURL(join(out, "refine.mjs")).href);
const f = JSON.parse(readFileSync("../tests/fixtures/refine_parity.json", "utf-8"));
const V = f.V.map((v) => Float32Array.from(v));
let bad = 0;
for (const detail of ["fine", "medium", "coarse"]) {
  const r = refineFile(V, f.keys, f.l1, detail);
  const e = f.expected[detail];
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  const ok = same(Array.from(r.labels), e.labels) && same(r.parent, e.parent) && same(r.representative, e.representative);
  console.log(`${detail}: ${ok ? "identical to Python" : "DIFFERENT"} (${r.sizes.length} groups)`);
  if (!ok) bad++;
}
process.exit(bad ? 1 : 0);
