
# MergeDNA Reproduction (with NanoChat Infrastructure Layer)

This repository contains a modular reproduction of the [**MergeDNA**](https://arxiv.org/abs/2511.14806) architecture using selected infrastructure components from [**NanoChat**](https://github.com/karpathy/nanochat) as a Transformer backbone and attention execution layer.

Rather than modifying NanoChat directly, this implementation wraps and extends NanoChat’s:
- Transformer encoder blocks
- FlashAttention (FA3 / SDPA) dispatch layer
- Distributed-training conventions
- Checkpointing / logging patterns

to support the hierarchical tokenisation and adaptive masking procedures required by MergeDNA.

---

## Documentation Map

- `report/MergeDNA_Implementation_Report.md`  
  Canonical description of design decisions and mapping to paper sections (4.1–4.5)

- `docs/infra_notes/`  
  Supporting infrastructure notes (e.g., SDPA/FA dispatch)

- `tests/`  
  Unit tests aligned to report sections

---

## Repository Structure

```
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
```

---

## How to Review this Submission

1. Read the implementation report:
   - `report/MergeDNA_Implementation_Report.md`

2. Run unit tests (inside Docker):

```bash
docker compose build
docker compose run --rm mergedna-dev pytest -q
```

3a. Run smoke pre-training (synthetic) using --tiny for speed:

```bash
docker compose run --rm mergedna-dev python scripts/pretrain_smoke.py --dataset synthetic --steps 50 --tiny
```
3b. Run smoke pre-training (synthetic):

```bash
docker compose run --rm mergedna-dev python scripts/pretrain_smoke.py --dataset synthetic --steps 50
```

Expected output:

```
step=10 loss_mtr=... loss_latent=... loss_amtm=...
```

This verifies that:
- MTR pass
- Latent-MTR pass (local encoder frozen)
- AMTM pass

execute end-to-end without error.

---

## Optional: HuggingFace Dataset Streaming

```bash
docker compose run --rm mergedna-dev python scripts/pretrain_smoke.py --dataset hf --hf-name InstaDeepAI/multi_species_genomes --steps 50
```

Tokenisation:
| Token | Meaning |
|-------|---------|
| A,C,G,T | canonical bases |
| N | ambiguity / unknown |
| [MASK] | AMTM input masking |

---

## Test ↔ Report Section Mapping

| Report Section | Test File |
|---------------|-----------|
| 4.1 - 4.2 Non-causal + Local Attention | `tests/test_41_42_attention_backend.py` |
| 4.3 Token Merge + Sparse S | `tests/test_43_local_merge_sparse_S.py` |
| 4.4 Latent Merge (L→K) | `tests/test_44_latent_merge_grouping.py` |
| 4.5 AMTM Sampler | `tests/test_45_amtm_sampler.py` |
| 4.5 AMTM Forward Contract | `tests/test_45_amtm_forward_contract.py` |
| 5.x Three-Pass Training | `tests/test_5xx_three_pass_training_contract.py` |

These tests collectively validate the architectural requirements described in Sections 4.1–4.5 of the implementation report.

---
## Optional: Logging & Checkpoiting Infrastructure
This reproduction includes optional training-infrastructure components adapted from the NanoChat execution layer:

- Structured JSON logging
- Throughput monitoring
- Checkpoint save / resume
- Weights & Biases experiment tracking (optional)

These components are not required to run the architectural reproduction described in Sections 4.1–4.5 of the implementation report.

By default:

- Logging is enabled locally
- Checkpointing is enabled via --ckpt-every
- W&B tracking is disabled unless explicitly requested

W&B can be enabled in:

- `offline mode` (local logging only)
- `online mode` (requires wandb install)

Example:
```docker compose run --rm mergedna-dev \
python scripts/pretrain_smoke.py \
--tiny \
--steps 50 \
--wandb offline
```
To enable W&B online tracking:

`uv sync --extra wandb`

If W&B is not installed, logging will be automatically disabled without affecting model execution.

These infrastructure components are provided solely for training observability and do not modify the MergeDNA modelling pipeline.

## Running Outside Docker (Optional)

```bash
pip install uv
uv sync
pytest -q
```

Docker execution is recommended to ensure matching dependency versions.

---

## Contact

Samuel Tonks  
Machine Learning Researcher
