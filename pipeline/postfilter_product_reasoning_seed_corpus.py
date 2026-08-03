"""Re-admit a frozen product-reasoning corpus against an expanded eval set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from build_product_reasoning_seed_corpus import (
    ProductCorpusError,
    load_eval_contamination,
    prompt_sha256,
    word_ngrams,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def postfilter(
    source: Path,
    eval_paths: list[Path],
    output: Path,
    report_path: Path,
) -> dict[str, object]:
    if output.exists():
        raise ProductCorpusError(f"refusing to replace corpus: {output}")
    if report_path.exists():
        raise ProductCorpusError(f"refusing to replace report: {report_path}")

    eval_exact, eval_ngrams = load_eval_contamination(eval_paths)
    source_sha256 = file_sha256(source)
    counters: Counter[str] = Counter()
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with source.open("r", encoding="utf-8") as reader, temporary.open("wb") as writer:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                counters["source_rows"] += 1
                row = json.loads(line)
                question = row.get("question")
                if not question:
                    raise ProductCorpusError(
                        f"source row {line_number} has no question field"
                    )
                identity = prompt_sha256(str(question))
                if identity in eval_exact:
                    counters["eval_exact_rejected"] += 1
                    continue
                if word_ngrams(str(question)) & eval_ngrams:
                    counters["eval_13gram_rejected"] += 1
                    continue
                encoded = json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                digest.update(encoded)
                writer.write(encoded)
                counters["admitted_rows"] += 1
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report: dict[str, object] = {
        "schema": "shohin-product-reasoning-corpus-postfilter-v1",
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "eval_paths": [str(path.resolve()) for path in eval_paths],
        "eval_prompt_count": len(eval_exact),
        "eval_13gram_count": len(eval_ngrams),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
        "counters": dict(sorted(counters.items())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = postfilter(args.source, args.eval, args.output, args.report)
    counters = report["counters"]
    assert isinstance(counters, dict)
    print(
        f"[product-corpus-postfilter] admitted={counters['admitted_rows']} "
        f"sha256={report['output_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
