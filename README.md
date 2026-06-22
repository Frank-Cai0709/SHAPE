# SHAPE: State-Aware HypergrAph concePt bottlEneck model for Interpretable Medical Image Classification

## Abstract

Concept bottleneck models (CBMs) provide an interpretable framework for medical image classifica- tion by predicting human-understandable concepts before diagnosis. However, existing CBMs often rely on weakly grounded concept discovery and model concepts as iso- lated, deterministic variables, limiting their ability to cap- ture clinically meaningful concepts, diagnostic uncertainty, and higher-order interactions. To address these limitations, we propose SHAPE, the State-Aware HypergrAph concePt bottlEneck model, for interpretable medical image classifi- cation. SHAPE models concepts from three complementary perspectives: discovery, uncertainty modeling, and struc- tured reasoning. Specifically, DACD exploits representative and boundary-sensitive exemplars to generate grounded medical concepts. SACM represents each concept as a distribution over diagnostic states, enabling uncertainty- aware concept modeling at the concept level rather than prediction level. DCR captures class-conditional diagnostic cue structures via hypergraph reasoning, enabling pre- dictions guided by structured concept combinations in- stead of independent concepts. Extensive experiments on six medical imaging datasets across nine classification settings show that SHAPE achieves competitive or supe- rior performance over recent CBM baselines while provid- ing more clinically meaningful explanations. These results demonstrate the effectiveness of concept-level uncertainty modeling and structured diagnostic reasoning for trustwor- thy medical image interpretation.

<p align="center">
  <img src="main_figure.png" width="900">
</p>

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
