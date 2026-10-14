# Jewelry Estimate Service

A FastAPI service that turns a jewelry brief plus an optional reference photo into
structured specifications, a ranked catalog match, an image-based part count, and a
confidence statement for every field.

**Domain:** fine jewelry — rings, chains, necklaces, bracelets, earrings, pendants.

The design goal is a service that knows when it is unsure. Roughly half of the
evaluation set is expected to go to human review, and every confidence level comes with
a written reason.

## Reviewer quickstart

```powershell
pip install -r requirements.txt
pytest            # 72 tests, no API key needed
python eval.py    # deterministic offline evaluation; no API key needed
uvicorn app.main:app --reload
```

That is the whole path **with no API key**. It runs the offline embedding backend, and
the service rebuilds its vector index automatically when it finds one built for a
different backend. To enable the Gemini paths, copy `.env.example` to `.env`, add your
key, then:

```powershell
python scripts\build_catalog_index.py --force
```

The demo references are owner-provided images; the iStock chain photo is redistributed
under the project owner's license. Catalog candidate images are real product photos
resolved from the CSV listing URLs, with licensed fallbacks for listings that block image
retrieval. Run `python scripts\generate_sample_assets.py` to regenerate only the demo
GIF and video. After analysis, the closest candidate image appears in its result card;
the uploaded reference photo is shown separately below.
Without a multimodal API key, OpenCV estimates image components but does not identify
jewelry type; image-only requests with an unknown type are not presented as catalog
recommendations.

## Run locally

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\fetch_inventory_images.py   # download catalog photos from CSV listing URLs
python scripts\generate_sample_assets.py  # optional: regenerate demo GIF/video only
python scripts\build_catalog_index.py      # one-time: describe + embed into Qdrant
$env:GEMINI_API_KEY = "your-key"           # optional; enables all Gemini paths
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the browser frontend, or `http://127.0.0.1:8000/docs`
for the API documentation.
The home page shows sample ring, chain, and pendant photos; click one to load it into the
reference-image field, then select **Analyze item** to see the uploaded photo and closest
candidate product image alongside the estimate.

### Demo media

- [Ring reference photo](examples/media/reference-ring.jpg)
- [Chain reference photo](examples/media/reference-chain.jpg)
- [Pendant reference photo](examples/media/reference-pendant.jpg)
- [Video walkthrough](examples/media/demo.mp4)

The photos are for reference-image demonstrations, not a price quote. Regenerate the
walkthrough GIF and video with `python scripts\generate_sample_assets.py`.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze `
  -ContentType 'application/json' `
  -Body '{"brief":"22K gold ruby ring weighing 4.1 grams"}'
```

With a photo:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/analyze `
  -ContentType 'application/json' `
  -Body '{"brief":"ruby ring in 22k gold","image":"data/images/RNG-002.jpg"}'
```

For a real multipart upload (supported directly by `/analyze`):
```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze `
  -F "brief=22K gold ruby ring" `
  -F "image=@reference.jpg;type=image/jpeg"
```
Uploads must be image files no larger than 5 MB. JSON requests with image data URLs or
server-side paths remain supported for existing clients and the evaluation harness.

Everything runs **without any API key**: extraction falls back to a conservative local
parser, and the vector index falls back to hashing text vectors plus an OpenCV image
descriptor. Adding `GEMINI_API_KEY` upgrades both. Nothing is hardcoded — every key and
model name is read from the environment.

## API key

Put it in a `.env` file in the project root — it is loaded automatically by the API, the
eval harness and both scripts, and `.env` is already gitignored.

```powershell
Copy-Item .env.example .env
notepad .env          # paste GEMINI_API_KEY=...
```

Only that one line is required:

```
GEMINI_API_KEY=your-key-here
```

Confirm it took:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "vector_index_ready": true,
  "llm_key_configured": true,
  "image_backend": "gemini",
  "gemini_model": "gemini-3.8-flash",
  "embedding_model": "gemini-embedding-2"
}
```

`llm_key_configured: true` and `image_backend: "gemini"` is what you want to see. The
endpoint never echoes the key itself.

**Then rebuild the index** — adding a key switches the embedding backend and changes the
vector width, so the index built without a key cannot be queried with one:

```powershell
python scripts\build_catalog_index.py --force
python eval.py --live-models
```

Until you do, `/search` returns a clear `used: false` telling you to rebuild rather than
failing with a dimension error. This step is also what turns the catalog descriptions from
restated CSV rows into real vision-model text.
The ordinary `python eval.py` command remains offline and does not consume provider quota;
`--live-models` explicitly enables configured models.

Other ways to supply it, in precedence order: a real environment variable beats `.env`
(so Docker and CI can inject secrets), and `.env` beats the built-in defaults.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/analyze` | Full pipeline: extract → retrieve → match → count → confidence |
| `POST` | `/search` | Pure vector retrieval, no deterministic re-rank |
| `GET` | `/inventory/{sku}` | Full catalog row, photo inlined as base64 |
| `GET` | `/inventory/{sku}/image` | The cached catalog photo as JPEG |
| `GET` | `/health` | Liveness plus vector-index readiness |

## Tests and evaluation

```powershell
pytest            # 72 tests
python eval.py    # 46 labelled cases; deterministic offline mode, no API key needed
python eval.py --verbose
python eval.py --live-models  # explicitly opt into configured model providers/quota
```

Metrics and the full failure analysis are in [EVALUATION.md](EVALUATION.md).

## How it works

```
catalog images + data/invent.csv
        │
        ▼  scripts/fetch_inventory_images.py
  data/images/<SKU>.jpg  +  manifest.json (sha, licence, image_is_unique)
        │
        ▼  scripts/build_catalog_index.py   (catalog_index.py)
  ┌───────────────────────── one-time ingest ─────────────────────────┐
  │  vision model  → "a polished yellow gold ring with a red stone"   │
  │  embed text + embed photo   (SAME multimodal model, one space)     │
  │  → Qdrant: one point per SKU, two named vectors                  │
  │      description · image       payload = full CSV row             │
  └───────────────────────────────────────────────────────────────────┘
        │
        ▼
brief + optional photo
        │
        ▼
  extractor.py       Gemini / OpenAI → Pydantic schema, retries → local rules
        │
        ├── specifications ──┐
        │                    ▼
        │            catalog_index.py   semantic recall from Qdrant (RRF fusion)
        │                    │            → semantic score per SKU
        │                    ▼
        │            matcher.py          weighted deterministic score
        │                                 (attributes + pixels + semantics)
        │                    │
        │                    ▼
        │              ranked candidates → product details + image
        │
        ├── photo ──► embeddings.py    image embeddings
        │             cv_estimator.py  contour count of small parts
        │             vision_verify.py Gemini re-check of the top 3 (advisory only)
        ▼
  review_engine.py   confidence per field + written reason + review flags
```

### The vector index

`catalog_index.py` ingests the catalog once:

1. Each `data/images/<SKU>.jpg` goes to a vision model, which writes a plain description.
   Cached by image hash in `data/index/descriptions.json`, so a VLM call is only paid when
   a photo actually changes.
2. The description text **and** the raw photo are embedded with the same multimodal model
   (`gemini-embedding-2`), so both land in one space. That is what lets a customer's photo
   and a customer's typed words query the same index.
3. Each SKU becomes one Qdrant point with two **named vectors** (`description` and
   `image`) and a payload holding the whole CSV row.

A query is embedded the same way and searched. A single signal keeps its own cosine
similarity; reciprocal rank fusion is used only when a brief *and* a photo are both
present, because rank fusion discards score margins and flattens results to
1.000 / 0.984 / 0.968 — exactly the discrimination the deterministic scorer needs.
Retrieval is **semantic**, but the deterministic scorer still decides the final ranking —
the vector store widens the candidate set, it does not get to choose a price.

**Visual similarity can never override a stated attribute.** A brief that says "ring" is
not answered with an earring, however close the embedding. `semantic_gate()` zeroes the
semantic contribution for any row contradicting an attribute the brief actually
determined, so the vector signal promotes a plausible product and never rescues a wrong
one.

### Value normalization

A model returns whatever phrasing the brief used — `"18k"`, `"sterling silver"`,
`"white gold"`, `"a purple stone"`, `"titanium"`. None of those are catalog values, so
they score 0 against every row while still reading like confident answers.
`app/normalize.py` snaps each field onto the vocabulary `data/invent.csv` actually uses,
or onto `unknown` when there is no honest mapping: `"white gold"` → `gold`,
`"purple stone"` → `amethyst`, `"titanium"` → `unknown`. Deterministic, runs after schema
validation, never invents a value the brief did not support.

Without a key the same pipeline runs on a 256-dim hashing text vector and the OpenCV image
descriptor, and every description is marked `source: "catalog_attributes"` so nobody
mistakes a restated CSV row for a real visual description.

### Deterministic matching

`matcher.py` scores every catalog row with fixed weights that sum to 1.0:

| Attribute | Weight | Score |
| --- | --- | --- |
| Semantic similarity | 0.25 | Qdrant score, normalized across the candidate set |
| Image similarity | 0.20 | min-max-normalized cosine between image embeddings |
| Item type | 0.18 | exact match |
| Metal | 0.15 | exact match |
| Purity | 0.09 | exact match |
| Gemstone | 0.08 | exact, or half credit for shared words ("black onyx" ~ "onyx") |
| Weight | 0.05 | linear, zero beyond a 5 g tolerance |

When a signal is unavailable its weight is redistributed proportionally across the ones
that remain, so a text-only request keeps the same relative priorities. Ties break on
ascending SKU, so equal scores never reorder between runs.

**Why the LLM does not pick the SKU.** A generated SKU cannot be recomputed by a human, is
not guaranteed to exist in the catalog, and can change between two identical requests. All
three are unacceptable when the output is a price. The vector store is a retrieval tool
here, not a decision-maker. See the module docstring in `matcher.py`.

### Vision verification is advisory only

`vision_verify.py` shows the vision model the customer's photo next to the shortlisted
catalog photos and asks only whether the product type and metal colour agree. Its answer
can **add a review flag**. It can never reorder the ranking or replace a SKU — that
would reintroduce exactly the non-determinism the matcher exists to prevent.

### Confidence and review flags

Every field returns `{"level": ..., "reason": ...}`. A level without a reason is
decoration, so the reason names the evidence: which model produced the value, whether
the photo was used, or which threshold was missed.

Review is triggered by any of:

- best match below 0.70
- **the brief omitted a quote-critical attribute (item type, metal, or purity), or
  determined fewer than 4 of the 5 attributes**, regardless of score — a high score must
  not disguise a guess or let the catalog fill in unstated product details
- an unreadable or corrupt photo
- the top candidate's catalog photo is shared with other SKUs, so the image cannot
  identify it
- the brief implies a part count and the CV count disagrees by more than one
- the vision model disputes the top candidate
- an implausible weight for the product type (e.g. a 900 g ring)

## Catalog images

`data/invent.csv` carries an `image_url` per SKU. These are usually **product page URLs**,
not direct image links, so the fetcher resolves them: it reads `og:image`, JSON-LD, and
raw `<img>` tags, filtering out icons, logos and sprites, then downloads what it finds.

Photos are cached to `data/images/<SKU>.jpg` and never fetched at request time, so the API
does not depend on a third-party host and embeddings stay cacheable.

```powershell
python scripts\fetch_inventory_images.py          # uses the cache
python scripts\fetch_inventory_images.py --force  # re-download
```

`data/images/manifest.json` records provenance for every file: the catalog URL, the
resolved image URL, SHA-256, byte size, attribution when available, and
`image_is_unique` with a `shared_with` list.

The current local cache contains **36 product photos** (26 retrieved from CSV listing
URLs and 10 replaced with the script's Wikimedia Commons fallbacks). It has 23 distinct
images because multiple SKUs in the CSV share listing URLs; those matches are flagged
for human review. Re-run `python scripts\fetch_inventory_images.py --force` to refresh
the cache and `python scripts\build_catalog_index.py --force` to rebuild image embeddings
after the photos change.

**To use your own product photos:** drop a file at `data/images/<SKU>.jpg` for each SKU and
re-run the fetcher, then rebuild the index. No code changes.

## Vector index

```powershell
python scripts\build_catalog_index.py          # incremental
python scripts\build_catalog_index.py --force  # rebuild the store from scratch
```

Descriptions and image embeddings are cached by content hash, so incremental runs are
cheap. Point IDs are derived from `sha256(sku)` — deliberately not Python's `hash()`, which
is salted per process and would silently duplicate the collection on every rebuild.

The Qdrant client runs **embedded on local disk** (`data/index/qdrant`), so there is no
container to run. One caveat worth knowing: the local backend locks its directory, so only
one client may exist per process; the module memoizes it.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | Enables Gemini extraction, embedding and verification |
| `LLM_PROVIDER` | `auto` | `auto`, `gemini`, or `openai` |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Extraction model |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.5-flash-lite` | Tried after the primary model fails |
| `GEMINI_DESCRIPTION_MODEL` | `gemini-3.8-flash` | Writes the catalog image descriptions |
| `GEMINI_VISION_MODEL` | `gemini-3.8-flash` | Verification model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Text **and** image embeddings |
| `GEMINI_EMBEDDING_DIMS` | `3072` | Vector width for the Gemini backend |
| `IMAGE_EMBEDDING_BACKEND` | `auto` | `auto`, `gemini`, or `opencv` |
| `INDEX_EMBEDDING_BACKEND` | follows the above | Embedding backend for the Qdrant index |
| `VISION_VERIFY_ENABLED` | `1` | Set `0` to skip verification entirely |
| `OPENAI_API_KEY` | unset | Optional alternate extraction provider |
| `CATALOG_INDEX_PATH` | `data/index/qdrant` | Optional local Qdrant storage path |
| `CATALOG_DESCRIPTION_CACHE` | `data/index/descriptions.json` | Optional description-cache path |

`gemini-2.5-flash` and `gemini-2.5-flash-lite` now return **404 for new accounts** and
`gemini-3.8-flash` intermittently answers 503/429, which is why the default chain carries
a second model rather than relying on retries.

## Free-tier quota

A free Gemini key allows roughly **20 `generate_content` requests per day**. The catalog
description pass needs one per image, so a 36-SKU catalog takes about two days to describe
fully. The evaluation harness does not use provider quota unless invoked with
`--live-models`.

The scripts are built for this. Descriptions are cached by image hash, so re-running
resumes where it stopped:

```powershell
python scripts\build_catalog_index.py                    # resume the description pass
python scripts\build_catalog_index.py --skip-descriptions  # re-embed, spend no quota
```

When quota runs out the build stops cleanly rather than writing filler, and reports what
is left:

```
STOPPED: the model provider refused further requests (daily free-tier quota).
Descriptions already written are kept. Re-run this script when the quota
resets to finish the remaining images - it resumes where it stopped.
```

Enabling billing on the key removes the limit entirely and is the obvious move if this
goes past a demo.

**Finish the description pass before trusting live semantic-retrieval results.** While it
is incomplete the index mixes visual prose with attribute restatements, which distorts
retrieval. The service says so rather than hiding it — `build_catalog_index.py` prints a
warning, and every `/analyze` response carries it in `warnings`:

```json
"warnings": ["catalog photo descriptions are incomplete, so semantic retrieval may be
 unreliable; re-run `python scripts/build_catalog_index.py` to finish it"]
```

Those are service-level conditions, deliberately kept out of `review_flags` — folding a
catalog-wide problem into per-request review would flag 100% of traffic and destroy the
signal that review flags are meant to carry.

The chain is `GEMINI_MODEL` → `GEMINI_FALLBACK_MODEL` → `OPENAI_MODEL` → local rules.
Every response reports which rung produced it in `extraction.source`, with
`fallback_used` and `fallback_reason` when degraded.

## Container

```powershell
docker compose up --build
```

## Deployment sketch on AWS

A Lambda container image behind API Gateway for the request path (low traffic, per-request
pricing fits a sales-assist tool); S3 for `data/images` with CloudFront in front if the
catalog grows past what fits in the image; DynamoDB for the review queue with an SNS topic
notifying sales when `needs_human_review` is true; CloudWatch for structured logs and the
cost/latency metrics noted below. Embeddings should be precomputed once per catalog
change and cached in S3 or ElastiCache, since they are already keyed by content hash —
a cold start should never pay for re-embedding the catalog. Secrets belong in Secrets
Manager referenced by the task definition, never in the image.

## Known limitations

- The catalog description pass may be incomplete because of the free-tier daily quota.
  Semantic retrieval is measurable but may be distorted by mixing visual descriptions
  with attribute restatements. Re-run `scripts/build_catalog_index.py` over a few days,
  or enable billing.
- Without `GEMINI_API_KEY` the index has no visual descriptions at all and text search is
  lexical rather than semantic.
- 8 groups of SKUs still share one catalog photo, so the image cannot identify a specific
  item within those groups.
- The count estimator merges touching parts (see `EVALUATION.md`, failure 1).
- `base_price` is the only price signal; there is no margin or setting-charge model.
- The 0.70 auto-select threshold is asserted, not calibrated against a labeled set.
- The embedded Qdrant store is fine at this catalog size but is single-process; a real
  deployment would point `QdrantClient` at a Qdrant server instead.
- `VECTOR_CANDIDATE_LIMIT` retrieves the whole catalog so every row gets a comparable
  semantic score. Past a few thousand rows this needs a real ANN index and an explicit
  "no semantic evidence" state.

## Use of AI tools

AI coding tools were used to help inspect the existing implementation, implement and test
the multipart upload path, and check the evaluation and documentation against the task.
Generated suggestions were treated as drafts: the final SKU decision remains a
deterministic weighted match rather than an LLM choice, malformed model output degrades
to labeled local rules, and low-confidence or conflicting results are routed for human
review. I also rejected presenting the prior live-provider metrics as reproducible after
they proved sensitive to API quota and index state; the evaluation now defaults to an
offline run, with live-model evaluation available only by explicit opt-in.

## Repository layout

```
app/            catalog_index (Qdrant ingest + retrieval), extractor, normalize,
                matcher, embeddings, cv_estimator, vision_verify,
                review_engine, schemas, config, main
data/           invent.csv (catalog), images/ (cached photos, gitignored),
                index/ (Qdrant store + description cache, gitignored)
eval_cases/     cases.json (46 labelled cases) + image fixtures
scripts/        fetch_inventory_images.py, build_catalog_index.py
eval.py         evaluation harness
tests/          72 pytest tests
```