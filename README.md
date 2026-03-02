# MergeDNA Reproduction (with NanoChat Infrastructure Layer)

This repository contains a clean, modular reproduction of the
**MergeDNA** architecture using selected infrastructure components from
the **NanoChat** repository as a Transformer backbone and attention
execution layer.

Rather than modifying NanoChat directly, this implementation wraps and
extends NanoChat's:

-   Transformer encoder blocks
-   FlashAttention (FA3 / SDPA) dispatch layer
-   Distributed-training conventions
-   Checkpointing / logging patterns

to support the hierarchical tokenisation and adaptive masking procedures
required by MergeDNA.

This preserves a separation between:

  Layer     |      Responsibility
  ----------| ----------------------------------------------
  NanoChat  | Transformer execution + attention backend
  MergeDNA  | Token merging, latent grouping, AMTM masking

The goal is to demonstrate a **modular integration approach** rather
than a from-scratch reimplementation of standard Transformer components.

------------------------------------------------------------------------

## Repository Structure

    mergedna/
      Local + latent encoders
      Adaptive Masked Token Modelling (AMTM)
      FlashAttention wrapper (non-causal + local)
      Global token grouping
      DNA vocab + tokenisation

    nanochat/
      Infrastructure layer:
      Transformer blocks
      FlashAttention backend dispatch
      RMSNorm + attention ops

    scripts/
      Smoke training script

    tests/
      Unit tests aligned to MergeDNA components

    report/
      MergeDNA_Implementation_Report.md

    docs/
      infra_notes


    Dockerfile
    docker-compose.yml

------------------------------------------------------------------------

## Quickstart (Recommended)

All tests and scripts are intended to run inside Docker for
reproducibility.

### 1. Build the development image

``` bash
docker compose build
```

------------------------------------------------------------------------

### 2. Run unit tests

``` bash
docker compose run --rm mergedna-dev pytest -q
```

This executes tests covering:

-   FlashAttention fallback (SDPA)
-   Local token merging
-   Latent token grouping
-   AMTM mask sampling
-   Mask projection to base resolution
-   Reconstruction contract tests

------------------------------------------------------------------------

### 3. Smoke Pre-Training Run

A short synthetic training run can be executed to validate:

-   three-pass training procedure
-   latent merge pathway
-   AMTM mask projection
-   reconstruction logits flow

Run:

``` bash
docker compose run --rm mergedna-dev \
python scripts/pretrain_smoke.py --dataset synthetic --steps 50
```

Expected output:

    step=10 loss_mtr=... loss_latent=... loss_amtm=...

This is not intended for convergence, but verifies that:

-   MTR pass
-   Latent-MTR pass (local encoder frozen)
-   AMTM pass

all execute end-to-end without error.

------------------------------------------------------------------------

## Optional: HuggingFace Dataset Streaming

To run the smoke script using real genomic data:

``` bash
docker compose run --rm mergedna-dev \
python scripts/pretrain_smoke.py \
--dataset hf \
--hf-name InstaDeepAI/multi_species_genomes \
--steps 50
```

DNA sequences are tokenised as:

  Token      Meaning
  ---------- ---------------------
  A,C,G,T    canonical bases
  N          ambiguity / unknown
  [MASK]   AMTM input masking

------------------------------------------------------------------------

## Implementation Notes

MergeDNA-specific functionality is implemented in wrappers within:

    mergedna/

These include:

  Section     | Component
  ----------- | ---------------------------------------
  4.1 / 4.2   | Non-causal + sliding-window attention
  4.3         | Local token merging
  4.4         | Global latent grouping
  4.5         | Adaptive Masked Token Modelling

NanoChat code remains largely unmodified and is treated as an execution
backend.

Unused NanoChat components (e.g. tokenizer, generation engine) are not
required for this reproduction and will be removed in a later cleanup
branch.

------------------------------------------------------------------------

## Running Outside Docker (Optional)

If required:

``` bash
pip install uv
uv sync
pytest -q
```

Docker execution is recommended to ensure matching dependency versions.

------------------------------------------------------------------------

## Contact

Samuel Tonks
Machine Learning Researcher
