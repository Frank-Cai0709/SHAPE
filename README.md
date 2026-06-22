# ECHO-CBM Core Release

This folder contains the minimal reviewer-facing core implementation of ECHO-CBM.

## Included modules

- `run_train.py`: main training entry
- `scripts/discover_concepts.py`: DACD concept discovery
- `utils/exemplar_selection.py`: representative/boundary exemplar selection
- `utils/diagnostic_evidence_distribution.py`: ECM / SACM module
- `utils/structured_diagnostic_cues.py`: SDCM module
- `utils/hypergraph_diagnostic_cues.py`: HDCR module
- `model/cbl.py`: concept bottleneck layer
- `glm_saga/elasticnet.py`: sparse linear optimization
- `data_utils.py`: dataset loading helpers
- `utils_my.py`: backbone/helper utilities
- `data/classes_name/`: dataset class-name files
- `requirements.txt`: package requirements

## Excluded contents

This release intentionally excludes:

- baseline implementations
- ablation-only scripts
- plotting / analysis tools
- checkpoints and saved outputs
- temporary experiment files

## Core pipeline

ECHO-CBM contains four main components:

1. DACD: dataset-adaptive concept discovery
2. ECM / SACM: evidential state-aware concept modeling
3. SDCM: structured diagnostic cue modeling
4. HDCR: hypergraph diagnostic cue reasoning

## Example usage

### 1. Discover concepts

```bash
python scripts/discover_concepts.py \
  --dataset HAM10000 \
  --split train \
  --sample_strategy representative_boundary \
  --num_exemplars 8
