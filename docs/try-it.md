# Try it on your own messages (early users)

Sphota groups a file of short texts (support tickets, reviews, survey answers, chat messages)
into **themes**, in any mix of languages, without uploading anything and without an API key.
The same complaint written in Hindi, Tamil, English or Spanish usually lands in the same theme.
This page takes about 10 minutes.

## 1. Get your file ready

* A **CSV** (or TSV, TXT with one message per line, or JSONL) with one message per row.
* A column named `text`, `message`, `body`, `comment`, `feedback` or `review` is picked up
  automatically (or choose the column yourself).
* Up to a few thousand rows is ideal. Remove names, phone numbers and emails if you can;
  nothing leaves your machine, but it is a good habit.

## 2a. In the browser (no install)

Open the Meaning Atlas, click **Analyze a file** (or drag the file onto the page), wait for
the encoder to load (~120 MB, once), and read the report on the right:

* each row is a **theme**, named after one of your own messages, with its count and share;
* click a theme to see example messages and the language mix, and to find it on the map;
* **Download results (CSV)** gives every message with its theme code (e.g. `WJX1`), region and
  full Sphota code.

Tick **exact codes** if you want the browser's codes to match the Python library exactly
(downloads a larger, ~470 MB model).

## 2b. On the command line

```bash
pip install "sphota[encode,cli] @ git+https://github.com/Thiruvikramanpravartak/sphota.git"
sphota analyze messages.csv -o report.html
```

Open `report.html` in a browser. The first run downloads the encoder (~470 MB; set
`SPHOTA_ENCODER=int8` for ~120 MB). `--by region` groups into 32 broad areas instead of themes;
`--min-count 10` hides small groups and sample texts (for sharing a report).

## 3. What to look for

* Do the biggest themes match what you know is in the data?
* Are messages about the same issue in different languages grouped together?
* Is anything grouped together that clearly should not be?

## 4. Tell us

Open a **[feedback issue](https://github.com/Thiruvikramanpravartak/sphota/issues/new?template=early-feedback.yml)**
(5 questions). Please describe private data rather than pasting it.

## Known limits

* Themes are the first block of the code: good for "what are people writing about", not for fine
  distinctions. Very short messages ("ok", "thanks") get low confidence.
* Codes are not reliable for **Santali (Ol Chiki)** and **Manipuri**; Assamese, Sindhi,
  Sanskrit and Kashmiri work only partly ([benchmark](benchmark.md)).
* Cell topic names come from Wikipedia and often do not fit everyday text, so they are hidden by
  default; themes are named by your own messages instead.
