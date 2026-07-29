#!/usr/bin/env python
"""Stream an HF dataset, QUALITY-FILTER + DECONTAMINATE, tokenize with the Shohin
tokenizer, and write zstd-compressed uint16 shards. Storage-lean: raw never lands.

Quality controls (master plan §6.5 — "highest quality possible"):
  --decontam-grams evalgrams.pkl  drop any doc containing an eval 13-gram
  --min-chars N                   drop trivially short docs
  --lang en --lang-field language keep only that language (where the field exists)

Writes a manifest.json with token counts and per-filter drop counts (audit trail).
vocab 32768 fits uint16. Shards: shard_NNNNN.u16.zst.

    python tokenize_shards.py --tokenizer tok.json --dataset HuggingFaceTB/finemath \\
        --config finemath-4plus --text-col text --lang en \\
        --decontam-grams evals/evalgrams.pkl --out-dir shards/finemath4 \\
        --shard-tokens 100000000 --max-tokens 4000000000
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import stat
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer


BOILERPLATE_MARKERS = (
    "accept cookies",
    "cookie policy",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "sign in",
    "log in",
    "subscribe to our newsletter",
    "javascript is disabled",
    "enable javascript",
)
DOCUMENT_LEDGER_NAME = "documents.jsonl.zst"
DOCUMENT_LEDGER_SCHEMA = "shohin-tokenized-document-v1"
IDENTITY_FIELDS = (
    "id",
    "blob_id",
    "content_id",
    "url",
    "repo_path",
    "repo_name",
    "commit_id",
    "path",
    "file_path",
    "source",
    "version",
)


def _grams(text, n):
    w = re.findall(r"\w+", text.lower())
    if w and len(w) < n:
        yield " ".join(w)
        return
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def direct_eval_grams(patterns, n):
    """Collect current eval-prompt grams, including prompts added after a pickle.

    Pretraining decontamination must not rely on a stale serialized evalgram
    set. This mirrors the SFT mix's direct prompt scan and covers short prompts
    by retaining their complete normalized word sequence.
    """
    result = set()
    fields = ("question", "problem", "prompt", "task", "text")
    for pattern in patterns or ():
        for path in sorted(glob.glob(pattern)):
            with open(path, errors="replace") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = next((str(row[field]) for field in fields if row.get(field)), "")
                    result.update(_grams(prompt, n))
    return result


def field_value(row, field):
    """Read a top-level or dotted nested dataset field without raising on drift."""
    if field in row:
        return row[field]
    value = row
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def exact_text_hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).digest()


def stable_document_identity(row, document_sha256):
    values = [
        f"{field}={row[field]}"
        for field in IDENTITY_FIELDS
        if row.get(field) not in (None, "")
    ]
    material = "\x1f".join(values) if values else document_sha256
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_receipt(path):
    candidate = Path(path)
    before_link = candidate.lstat()
    if candidate.is_symlink() or not stat.S_ISREG(before_link.st_mode):
        raise RuntimeError(f"input is not a regular non-symlink file: {candidate}")
    resolved = candidate.resolve()
    before = resolved.stat()
    if before.st_nlink != 1:
        raise RuntimeError(f"input is not a single-link file: {resolved}")
    digest = sha256_file(resolved)
    after = resolved.stat()
    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_nlink,
        )
    if identity(before) != identity(after):
        raise RuntimeError(f"input changed while being measured: {resolved}")
    return {
        "path": str(resolved),
        "bytes": before.st_size,
        "sha256": digest,
    }


def verify_file_receipt(receipt):
    current = file_receipt(receipt["path"])
    if current != receipt:
        raise RuntimeError(f"input file changed during tokenization: {receipt['path']}")


def resolve_local_inputs(paths, revision):
    if paths is None:
        return None, []
    if not revision:
        raise ValueError("--input-files requires an explicit --revision")
    resolved = sorted(str(Path(path).resolve()) for path in paths)
    if len(resolved) != len(set(resolved)):
        raise ValueError("--input-files contains duplicate paths")
    return resolved, [file_receipt(path) for path in resolved]


def local_input_format(paths):
    if paths is None:
        return None
    formats = set()
    for path in paths:
        name = Path(path).name.lower()
        if name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
            formats.add("json")
        elif name.endswith(".parquet"):
            formats.add("parquet")
        else:
            raise ValueError(f"unsupported local input format: {path}")
    if len(formats) != 1:
        raise ValueError("--input-files must use one homogeneous file format")
    return formats.pop()


def canonical_payload_sha256(payload):
    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


class DocumentLedgerWriter:
    """Write a compressed, text-free document-to-token provenance ledger."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows = 0
        self.tokens = 0
        self._raw = self.path.open("xb")
        self._stream = zstd.ZstdCompressor(level=3).stream_writer(
            self._raw,
            closefd=False,
        )
        self._closed = False

    def write(self, record):
        if self._closed:
            raise RuntimeError("document ledger is already closed")
        payload = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        self._stream.write(payload)
        self.rows += 1
        self.tokens += int(record["tokens"])

    def close(self):
        if self._closed:
            raise RuntimeError("document ledger is already closed")
        self._stream.flush(zstd.FLUSH_FRAME)
        self._stream.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._closed = True
        receipt = file_receipt(self.path)
        return {
            "path": self.path.name,
            "bytes": receipt["bytes"],
            "sha256": receipt["sha256"],
            "rows": self.rows,
            "tokens": self.tokens,
            "contains_document_text": False,
            "schema": DOCUMENT_LEDGER_SCHEMA,
        }


def max_line_repeat_fraction(text):
    lines = [" ".join(line.split()).lower() for line in text.splitlines() if line.strip()]
    if not lines:
        return 1.0
    counts = Counter(lines)
    return max(counts.values()) / len(lines)


def boilerplate_marker_count(text):
    lowered = text.lower()
    return sum(marker in lowered for marker in BOILERPLATE_MARKERS)


def extraction_quality(text):
    denominator = max(len(text), 1)
    return {
        "alpha_fraction": sum(character.isalpha() for character in text)
        / denominator,
        "control_fraction": sum(
            ord(character) < 32 and character not in "\n\r\t"
            for character in text
        )
        / denominator,
        "replacement_fraction": text.count("\ufffd") / denominator,
    }


def domain_value(row, field):
    raw = field_value(row, field)
    if not raw:
        return "<missing>"
    value = str(raw)
    if "://" in value:
        return (urlparse(value).hostname or "<missing>").lower()
    return value.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "physical local JSON/JSON.GZ source files; bypasses remote dataset "
            "resolution and requires an explicit --revision"
        ),
    )
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--text-cols", nargs="+", default=None,
                    help="concat multiple fields (joined by a blank line) instead of --text-col; "
                         "e.g. --text-cols problem generated_solution for OpenMathInstruct-2. "
                         "Decontam/min-chars run on the concatenated text.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--decontam-grams", default=None)
    ap.add_argument("--eval-glob", nargs="*", default=[],
                    help="live eval JSONL globs whose prompt n-grams augment --decontam-grams")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--exact-dedup", action="store_true")
    ap.add_argument(
        "--max-line-repeat-fraction",
        type=float,
        default=1.0,
        help="reject documents whose most repeated normalized line exceeds this fraction",
    )
    ap.add_argument(
        "--max-boilerplate-markers",
        type=int,
        default=-1,
        help="-1 = unlimited",
    )
    ap.add_argument(
        "--min-alpha-fraction",
        type=float,
        default=0.0,
        help="reject extraction with a lower alphabetic-character fraction",
    )
    ap.add_argument(
        "--max-control-fraction",
        type=float,
        default=1.0,
        help="reject extraction with a higher non-whitespace control fraction",
    )
    ap.add_argument(
        "--max-replacement-fraction",
        type=float,
        default=1.0,
        help="reject extraction with a higher Unicode replacement-character fraction",
    )
    ap.add_argument("--allowed-values-field", default=None)
    ap.add_argument(
        "--allowed-values",
        nargs="+",
        default=None,
        help="exact allowed values for --allowed-values-field",
    )
    ap.add_argument("--domain-field", default=None)
    ap.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=0,
        help="0 = unlimited; applied after tokenization",
    )
    ap.add_argument("--lang", default=None)
    ap.add_argument("--lang-field", default="language")
    ap.add_argument(
        "--require-lang-field",
        action="store_true",
        help="when --lang is set, reject rather than accept rows with missing language metadata",
    )
    ap.add_argument("--min-number-field", default=None,
                    help="optional numeric quality field (supports dotted paths); drop a row when missing or below --min-number")
    ap.add_argument("--min-number", type=float, default=None,
                    help="minimum accepted value for --min-number-field")
    a = ap.parse_args()

    if os.path.lexists(a.out_dir):
        if os.path.islink(a.out_dir) or not os.path.isdir(a.out_dir) or os.listdir(a.out_dir):
            raise FileExistsError(f"refusing nonempty or non-directory output: {a.out_dir}")
    else:
        os.makedirs(a.out_dir)
    tok = Tokenizer.from_file(a.tokenizer)
    assert tok.get_vocab_size() <= 65535, "vocab exceeds uint16"
    eos_id = tok.token_to_id(a.eos)
    if (a.min_number_field is None) != (a.min_number is None):
        raise ValueError("--min-number-field and --min-number must be provided together")
    if a.max_chars and a.max_chars < a.min_chars:
        raise ValueError("--max-chars must be zero or at least --min-chars")
    if not 0 < a.max_line_repeat_fraction <= 1:
        raise ValueError("--max-line-repeat-fraction must be in (0, 1]")
    if (a.domain_field is None) != (a.max_tokens_per_domain == 0):
        raise ValueError(
            "--domain-field and positive --max-tokens-per-domain must be provided together"
        )
    if a.require_lang_field and not a.lang:
        raise ValueError("--require-lang-field requires --lang")
    if (a.allowed_values_field is None) != (a.allowed_values is None):
        raise ValueError(
            "--allowed-values-field and --allowed-values must be provided together"
        )
    if not 0 <= a.min_alpha_fraction <= 1:
        raise ValueError("--min-alpha-fraction must be in [0, 1]")
    if not 0 <= a.max_control_fraction <= 1:
        raise ValueError("--max-control-fraction must be in [0, 1]")
    if not 0 <= a.max_replacement_fraction <= 1:
        raise ValueError("--max-replacement-fraction must be in [0, 1]")
    allowed_values = (
        set(str(value) for value in a.allowed_values)
        if a.allowed_values is not None
        else None
    )

    S = gram_n = None
    pickle_gram_count = direct_gram_count = 0
    if a.decontam_grams:
        d = pickle.load(open(a.decontam_grams, "rb"))
        S, gram_n = set(d["grams"]), d["n"]
        pickle_gram_count = len(S)
    if a.eval_glob:
        gram_n = gram_n or 13
        direct = direct_eval_grams(a.eval_glob, gram_n)
        direct_gram_count = len(direct)
        S = set(S or ())
        S.update(direct)

    selection_code_receipt = file_receipt(__file__)
    tokenizer_receipt = file_receipt(a.tokenizer)
    pickle_receipt = file_receipt(a.decontam_grams) if a.decontam_grams else None
    eval_files = []
    seen_eval_paths = set()
    for pattern in a.eval_glob:
        for path in sorted(glob.glob(pattern)):
            absolute = str(Path(path).resolve())
            if absolute not in seen_eval_paths:
                eval_files.append(file_receipt(absolute))
                seen_eval_paths.add(absolute)

    input_files, input_file_receipts = resolve_local_inputs(
        a.input_files,
        a.revision,
    )
    input_format = local_input_format(input_files)
    if a.input_files is not None:
        assert input_files is not None
        assert input_format is not None
        resolved_revision = a.revision
        ds = load_dataset(
            input_format,
            data_files=input_files,
            split=a.split,
            streaming=True,
        )
    else:
        resolved_revision = a.revision or HfApi().dataset_info(a.dataset).sha
        kw = dict(split=a.split, streaming=True, revision=resolved_revision)
        if a.config:
            kw["name"] = a.config
        ds = load_dataset(a.dataset, **kw)

    cctx = zstd.ZstdCompressor(level=3)
    document_ledger = DocumentLedgerWriter(
        Path(a.out_dir) / DOCUMENT_LEDGER_NAME
    )
    buf, shard, tok_total = [], 0, 0
    shard_files = []
    seen_hashes = set()
    domain_tokens = Counter()
    seen = kept = n_short = n_long = n_lang = n_lang_missing = n_quality = n_contam = 0
    n_duplicate = n_repetition = n_boilerplate = n_domain_cap = 0
    n_extraction_quality = n_allowed_value = 0

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        p = os.path.join(a.out_dir, f"shard_{shard:05d}.u16.zst")
        compressed = cctx.compress(arr.tobytes())
        with open(p, "wb") as f:
            f.write(compressed)
        shard_files.append(
            {
                "path": os.path.basename(p),
                "bytes": len(compressed),
                "tokens": int(len(arr)),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        )
        print(f"[shard] {p} {len(arr):,} tok {os.path.getsize(p)/1e6:.1f}MB", flush=True)
        shard += 1
        buf = []

    for ex in ds:
        seen += 1
        if a.text_cols:
            parts = [str(ex.get(c) or "") for c in a.text_cols]
            txt = "\n\n".join(p for p in parts if p)
        else:
            txt = ex.get(a.text_col) or ""
        if len(txt) < a.min_chars:
            n_short += 1
            continue
        if a.max_chars and len(txt) > a.max_chars:
            n_long += 1
            continue
        if max_line_repeat_fraction(txt) > a.max_line_repeat_fraction:
            n_repetition += 1
            continue
        if (
            a.max_boilerplate_markers >= 0
            and boilerplate_marker_count(txt) > a.max_boilerplate_markers
        ):
            n_boilerplate += 1
            continue
        quality = extraction_quality(txt)
        if (
            quality["alpha_fraction"] < a.min_alpha_fraction
            or quality["control_fraction"] > a.max_control_fraction
            or quality["replacement_fraction"] > a.max_replacement_fraction
        ):
            n_extraction_quality += 1
            continue
        if allowed_values is not None:
            selected_value = field_value(ex, a.allowed_values_field)
            if str(selected_value) not in allowed_values:
                n_allowed_value += 1
                continue
        if a.lang:
            lv = field_value(ex, a.lang_field)
            if lv is None and a.require_lang_field:
                n_lang_missing += 1
                continue
            if lv is not None and str(lv).lower() != a.lang.lower():
                n_lang += 1
                continue
        if a.min_number_field is not None:
            try:
                score = float(field_value(ex, a.min_number_field))
            except (TypeError, ValueError):
                n_quality += 1
                continue
            if score < a.min_number:
                n_quality += 1
                continue
        document_sha256 = exact_text_hash(txt).hex()
        digest = bytes.fromhex(document_sha256) if a.exact_dedup else None
        if digest is not None:
            if digest in seen_hashes:
                n_duplicate += 1
                continue
        if S is not None and any(g in S for g in _grams(txt, gram_n)):
            n_contam += 1
            continue
        ids = tok.encode(txt).ids
        document_token_values = [
            *ids,
            *([eos_id] if eos_id is not None else []),
        ]
        document_tokens = len(document_token_values)
        token_sha256 = hashlib.sha256(
            np.asarray(document_token_values, dtype=np.uint16).tobytes()
        ).hexdigest()
        domain = domain_value(ex, a.domain_field) if a.domain_field else None
        if (
            domain is not None
            and domain_tokens[domain] + document_tokens > a.max_tokens_per_domain
        ):
            n_domain_cap += 1
            continue
        if digest is not None:
            seen_hashes.add(digest)
        token_start = len(buf)
        buf.extend(document_token_values)
        document_ledger.write(
            {
                "schema": DOCUMENT_LEDGER_SCHEMA,
                "source_row_index": seen - 1,
                "stable_identity_sha256": stable_document_identity(
                    ex,
                    document_sha256,
                ),
                "document_sha256": document_sha256,
                "domain": domain,
                "allowed_value": (
                    str(field_value(ex, a.allowed_values_field))
                    if a.allowed_values_field
                    else None
                ),
                "chars": len(txt),
                "tokens": document_tokens,
                "shard": f"shard_{shard:05d}.u16.zst",
                "token_start": token_start,
                "token_end": token_start + document_tokens,
                "token_sha256": token_sha256,
            }
        )
        tok_total += document_tokens
        if domain is not None:
            domain_tokens[domain] += document_tokens
        kept += 1
        if len(buf) >= a.shard_tokens:
            flush()
        if a.max_tokens and tok_total >= a.max_tokens:
            break
    flush()
    document_ledger_receipt = document_ledger.close()

    if sum(record["tokens"] for record in shard_files) != tok_total:
        raise RuntimeError("shard token ledger does not match retained token total")
    if (
        document_ledger_receipt["rows"] != kept
        or document_ledger_receipt["tokens"] != tok_total
    ):
        raise RuntimeError("document ledger does not reconcile with retained corpus")
    verify_file_receipt(selection_code_receipt)
    verify_file_receipt(tokenizer_receipt)
    if pickle_receipt is not None:
        verify_file_receipt(pickle_receipt)
    for receipt in eval_files:
        verify_file_receipt(receipt)
    for receipt in input_file_receipts:
        verify_file_receipt(receipt)
    manifest = {
        "schema": "shohin-tokenized-shards-v3",
        "dataset": a.dataset,
        "config": a.config,
        "split": a.split,
        "requested_revision": a.revision,
        "resolved_revision": resolved_revision,
        "local_input_format": input_format,
        "source_files": input_file_receipts,
        "selection_code_sha256": selection_code_receipt["sha256"],
        "tokenizer": {
            **tokenizer_receipt,
            "vocab_size": tok.get_vocab_size(),
            "eos_token": a.eos,
            "eos_id": eos_id,
        },
        "tokens": tok_total,
        "shards": shard,
        "shard_files": shard_files,
        "document_ledger": document_ledger_receipt,
        "seen": seen,
        "kept": kept,
        "dropped_short": n_short,
        "dropped_long": n_long,
        "dropped_lang": n_lang,
        "dropped_language_missing": n_lang_missing,
        "dropped_quality": n_quality,
        "dropped_duplicate": n_duplicate,
        "dropped_repetition": n_repetition,
        "dropped_boilerplate": n_boilerplate,
        "dropped_extraction_quality": n_extraction_quality,
        "dropped_allowed_value": n_allowed_value,
        "dropped_domain_cap": n_domain_cap,
        "dropped_contam": n_contam,
        "decontamination": {
            "gram_n": gram_n,
            "pickle_grams": pickle_gram_count,
            "pickle_path": pickle_receipt["path"] if pickle_receipt else None,
            "pickle_bytes": pickle_receipt["bytes"] if pickle_receipt else None,
            "pickle_sha256": pickle_receipt["sha256"] if pickle_receipt else None,
            "direct_eval_grams": direct_gram_count,
            "eval_globs": a.eval_glob,
            "eval_files": eval_files,
        },
        "filters": {
            "text_col": a.text_col,
            "text_cols": a.text_cols,
            "min_chars": a.min_chars,
            "max_chars": a.max_chars,
            "language": a.lang,
            "language_field": a.lang_field,
            "require_language_field": a.require_lang_field,
            "minimum_number_field": a.min_number_field,
            "minimum_number": a.min_number,
            "exact_dedup": a.exact_dedup,
            "max_line_repeat_fraction": a.max_line_repeat_fraction,
            "max_boilerplate_markers": a.max_boilerplate_markers,
            "min_alpha_fraction": a.min_alpha_fraction,
            "max_control_fraction": a.max_control_fraction,
            "max_replacement_fraction": a.max_replacement_fraction,
            "allowed_values_field": a.allowed_values_field,
            "allowed_values": (
                sorted(allowed_values) if allowed_values is not None else None
            ),
            "domain_field": a.domain_field,
            "max_tokens_per_domain": a.max_tokens_per_domain,
            "retained_domains": len(domain_tokens),
        },
    }
    manifest["payload_sha256"] = canonical_payload_sha256(manifest)
    manifest_path = os.path.join(a.out_dir, "manifest.json")
    with open(manifest_path, "x") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print("[done]", json.dumps(manifest))


if __name__ == "__main__":
    main()
