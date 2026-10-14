"""Build the vector index over data/invent.csv and its cached catalog photos.

Run this after changing the catalog or adding product photos. Descriptions are cached by
image hash, so re-running is cheap and only pays for images that actually changed.

    python scripts/build_catalog_index.py            # incremental
    python scripts/build_catalog_index.py --force    # rebuild everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog_index import QuotaExhausted, build_index, get_product  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild the whole index")
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="embed what is already cached, spending no generation quota",
    )
    parser.add_argument("--show", metavar="SKU", help="print one indexed product")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        summary = build_index(force=args.force, describe=not args.skip_descriptions)
    except QuotaExhausted as exc:
        print(
            "\nSTOPPED: the model provider refused further requests (daily free-tier quota).\n"
            f"        {exc}\n\n"
            "Descriptions already written are kept. Re-run this script when the quota\n"
            "resets to finish the remaining images - it resumes where it stopped.",
            file=sys.stderr,
        )
        return 2

    print(
        "\nindexed {indexed} SKUs into {collection} (backend={backend}, "
        "dims={dimensions})".format(**summary)
    )
    print(
        "descriptions: {descriptions_vlm} from the vision model, "
        "{descriptions_from_attributes} from catalog attributes, "
        "{descriptions_pending_quota} waiting on quota".format(**summary)
    )
    if summary.get("description_sources_mixed"):
        print(
            "\nWARNING: description sources are inconsistent. A partially described catalog\n"
            "         embeds a few rows whose text has a completely different shape, which\n"
            "         distorts retrieval. Re-run this script (it resumes) until every image\n"
            "         has a vision-model description."
        )
    if summary["skipped"]:
        print(f"not indexed yet: {', '.join(summary['skipped'])}")
    print(f"stored at: {summary['path']}")

    if args.show:
        print(f"\n{get_product(args.show, include_image=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())