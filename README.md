
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
  This report is a detailed description of design decisions (sections 4.1–4.5,5.0) that map to the information provided about the MergeDNA implementation in the paper. 

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

## Quickstart

1. Read the implementation report:
   - `report/MergeDNA_Implementation_Report.md`

2. Run unit tests (inside Docker):

```bash
docker compose build
docker compose run --rm mergedna-dev pytest -v
```
---

3. Optional: Running Outside Docker

```bash
pip install uv
uv sync
pytest -v
```

Docker execution is recommended to ensure matching dependency versions.

---


4. Run smoke pre-training (synthetic) using --tiny for speed:

```bash
docker compose run --rm mergedna-dev python scripts/pretrain_smoke.py --dataset synthetic --steps 50 --tiny
```
5. Run smoke pre-training (synthetic) (default: larger sequence length and model width):

```bash
docker compose run --rm mergedna-dev python scripts/pretrain_smoke.py --dataset synthetic --steps 50
```

Expected output:

```
step=10 loss_mtr=... loss_latent=... loss_amtm=...
```

This verifies that:
- MTR pass
- Latent-MTR pass
- AMTM pass execute end-to-end without error
- Three forward passes execute successfully


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
| 4.3 Layerwise Merging | `tests/test_43_layerwise_merge_schedule.py` |
| 4.4 Latent Merge (L→K) | `tests/test_44_latent_merge_grouping.py` |
| 4.5 AMTM Sampler | `tests/test_45_amtm_sampler.py` |
| 4.5 AMTM Forward Contract | `tests/test_45_amtm_forward_contract.py` |
| 5.x Three-Pass Training | `tests/test_5xx_three_pass_training_contract.py` |

These tests collectively validate the architectural requirements described in Sections 4.1–4.5 and 5.0 of the implementation report.


## Short Architectural Overview
```
Base Tokens (N)
   ↓ (4.1 / 4.2)
Local Encoder + Token Merge (4.3)
   ↓  N → L
Latent Encoder + Global Merge (4.4)
   ↓  L → K
Latent Decoder
   ↓
Local Decoder + Unmerge (4.3)
   ↓
Base Logits (N)

AMTM (4.5): Mask sampling via 1/g_i^2 from latent grouping.
```
In addition to structural unit tests, this implementation includes behavioural diagnostics aligned explicitly to the architectural components described in Sections 4.1 - 4.5 of the report.

## Contact

Samuel Tonks  
Machine Learning Researcher
