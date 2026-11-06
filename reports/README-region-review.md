# Reviewing the 32 region names

`region_names_review.csv` opens in Excel or Google Sheets (File → Import). One row per region.

For each language you speak (Hindi, Tamil, Bengali):
1. Read the English name and the example topics, so you know what the region is about.
2. Read the name in your language. Put **yes** in "ok?" if it is natural and correct.
3. If not, put **no** and write a better short name (1–4 words) in "correction".
4. Where it says "(missing - please write one)", write a name.

Then send the file back (or commit it). The corrections go into `src/sphota/data/regions/regions.json`
(`labels_i18n`). Names are informative only, so changing them never changes anyone's codes.
