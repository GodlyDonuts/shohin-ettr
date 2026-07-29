from pathlib import Path

import numpy as np
import zstandard as zstd

from data import ShardLoader


def test_cpu_batch_does_not_require_pinned_accelerator_memory(tmp_path: Path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    tokens = np.arange(128, dtype=np.uint16)
    (shard_dir / "shard_00000.u16.zst").write_bytes(
        zstd.ZstdCompressor().compress(tokens.tobytes())
    )
    loader = ShardLoader(
        [str(shard_dir)],
        seq_len=8,
        batch_size=2,
        prefetch=1,
    )
    x, y = loader.next_batch("cpu")
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
