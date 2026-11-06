// Encode the bundled sample in the built Atlas (vite preview must be running) and print its codes.
//   npm run build && npm run preview & node tests/sample-codes.mjs [url] > codes.json
// Used to check that dependency upgrades leave in-browser codes unchanged, and that the
// "Analyze a file" flow works end to end without page errors.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const url = process.argv[2] || "http://localhost:4173/";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(url, { waitUntil: "networkidle" });
await page.click("#sample-file");
await page.waitForSelector(".report .regions li", { timeout: 300000 });
const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#dl-file")]);
const csv = readFileSync(await dl.path(), "utf-8");
await browser.close();
const codes = csv.split(/\r\n/).slice(1).filter(Boolean).map((l) => l.match(/([0-9A-Z]{4}-[0-9A-Z]{2}-[0-9A-Z]{2}-[0-9A-Z]{2})/)?.[1]);
if (errors.length) { console.error("page errors:", errors); process.exit(1); }
console.log(JSON.stringify(codes));
