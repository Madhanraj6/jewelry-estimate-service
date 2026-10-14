# Evaluation

Run `python eval.py` for a deterministic, offline evaluation; use `--verbose` for
per-case results. This run covers 46 labeled cases: 15 text-only briefs, 10 component
count images, 8 brief-plus-image requests, 5 image-only requests, and 8 deliberately
ambiguous briefs. The offline harness uses local extraction rules, hashing text
embeddings, OpenCV image descriptors, and a temporary Qdrant index. It does not read
cached model descriptions or call a provider. `python eval.py --live-models` opts into
configured providers and may use API quota or incur cost.

## Metrics

Measured with `python eval.py` on 2026-10-14:

| Metric | Result | Evaluation subset |
| --- | ---: | --- |
| Extraction field accuracy | **99.1%** (114/115 fields) | 23 requests with an expected SKU × 5 fields |
| Matching top-1 hit rate | **100.0%** (23/23) | Requests with an expected SKU |
| Matching top-3 hit rate | **100.0%** (23/23) | Requests with an expected SKU |
| Image count MAE | **0.60** | 10 images with a labeled component count |
| Review-flag rate | **60.9%** (28/46) | All requests |
| Vector recall of expected SKU | **100.0%** (23/23) | Expected SKU appears in retrieval top 10 |
| Brief-plus-image top-1 hit rate | **100.0%** (8/8) | Brief and reference photo supplied |
| Image-only item-type accuracy | **60.0%** (3/5) | No text supplied |
| Image-only review rate | **100.0%** (5/5) | No text supplied |
| Review recall on labeled review cases | **100.0%** (8/8) | The 8 cases labeled `expect_review: true` |
| Review precision | **Not measurable** | The set has no labeled non-review cases |

Extraction accuracy is computed only for the 23 cases with a known expected SKU; it is
not an estimate of accuracy on arbitrary briefs. The briefs repeat catalog attributes
and often include exact weights. Similarly, the eight brief-plus-image cases use their
associated synthetic catalog illustration as the reference, not a new customer
photograph. Their perfect match rate checks the image pipeline and deterministic
ranking, not robustness to real-world photography. The image-only result is a more
honest signal of the current visual limitation.

The review precision metric is intentionally reported as unavailable: all eight cases
with review labels are positive examples, so false-positive performance cannot be
measured. The overall review-flag rate is descriptive, not a precision or quality score.

## Failure analysis

The three largest count-estimation errors in this set are:

### 1. Touching components merge (`count-hard-01`)

Expected five; OpenCV counted one (absolute error 4). The baseline finds external
contours, so overlapping circles form one connected foreground region. Increasing the
minimum-area filter cannot split a merged contour.

**Next change:** add a distance-transform watershed or a circle detector as a second
estimator, and only accept a split when the methods agree. Until then, treat the count as
advisory and keep image/text disagreements on the review path.

### 2. Uneven background affects the global threshold (`count-hard-02`)

Expected four; OpenCV counted three (absolute error 1). Otsu thresholding chooses one
threshold for the whole frame, but this fixture has a left-to-right intensity gradient.
Part of a foreground object can blend into the background and disappear as a separate
contour.

**Next change:** evaluate adaptive thresholding and background normalization against
fixtures with controlled illumination gradients before selecting a replacement.

### 3. Debris creates a false component (`count-hard-03`)

Expected three; OpenCV counted four (absolute error 1). Small foreground specks survive
thresholding and the fixed area cutoff, so at least one is counted as a component.

**Next change:** combine area with shape and size consistency, and evaluate those
thresholds on additional labeled examples so filtering debris does not discard small
real stones.

## Other observed limitations

- Two of five image-only cases have the wrong item type. With no configured VLM, the
  local extraction fallback does not classify jewelry type from the image. The service
  holds those requests for review instead of treating OpenCV similarity as a classifier.
- The service has not been evaluated on real customer photographs. Synthetic count
  fixtures and synthetic catalog illustrations are limited tests.
- The 0.70 auto-select threshold is a policy choice, not a statistically calibrated
  operating point. More positive and negative review labels are needed to tune it.
- The offline extraction path is a conservative local parser, not an LLM. The
  malformed-output/retry fallback behavior is covered by pytest mocks; live-provider
  extraction quality is not measured by the offline metrics above.
- Some SKUs intentionally share a synthetic illustration because their original source
  catalog URL was shared. The review policy downgrades matches where the image cannot
  distinguish those SKUs.

## Next steps

1. Collect distinct, representative customer and catalog photos and label them
   independently of the catalog rows.
2. Add labeled examples that should *not* require review, then report review precision
   and recall on a balanced set.
3. Improve the count baseline for touching objects, uneven lighting, and debris; retain
   a review threshold until performance is measured on real images.
4. Add paraphrases, typos, units, and intentionally missing attributes to the brief
   set; avoid serializing the exact inventory row as the test prompt.
5. Sweep and calibrate the auto-select threshold using a held-out set.
6. Run `python eval.py --live-models` separately when provider credentials are available,
   record the model names and date, and keep those results separate from the reproducible
   offline baseline.
