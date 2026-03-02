from __future__ import annotations
import random
import torch
from datasets import load_dataset

from mergedna.dna_vocab import encode_dna_string


def make_hf_dna_dataloader(
    dataset_name: str,
    seq_len: int,
    device_batch_size: int,
    split: str = "train",
    seed: int = 1234,
    *,
    trust_remote_code: bool = True,
    shuffle_buffer: int | None = 10_000,
    start_bias_first_200: bool = True,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Streaming generator yielding (device_batch_size, seq_len) int64 tokens.

    Notes:
      - Uses the dataset's canonical `sequence` field (as per HF dataset card).
      - Optionally biases random crop start into first 200 positions (dataset-card training trick).
      - Supports simple DDP sharding via rank/world_size.
    """
    rng = random.Random(seed + rank)

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        trust_remote_code=trust_remote_code,
    )

    if shuffle_buffer is not None and shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    # Prefer dataset-native sharding if available
    if world_size > 1 and hasattr(ds, "shard"):
        ds = ds.shard(num_shards=world_size, index=rank)

    it = iter(ds)

    def _sample_start(seq: str) -> int:
        max_start = len(seq) - seq_len
        if max_start <= 0:
            return 0
        if start_bias_first_200:
            return rng.randint(0, min(200, max_start))
        return rng.randint(0, max_start)

    while True:
        batch = []
        while len(batch) < device_batch_size:
            try:
                row = next(it)
            except StopIteration:
                it = iter(ds)
                continue

            # Use canonical field name
            seq = row.get("sequence", None)
            if not isinstance(seq, str) or len(seq) < seq_len:
                continue

            start = _sample_start(seq)
            chunk = seq[start : start + seq_len]
            tok = encode_dna_string(chunk)
            batch.append(tok)

        yield torch.tensor(batch, dtype=torch.long)