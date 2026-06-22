# State-Aware HypergrAph concePt bottlEneck model for Interpretable Medical Image Classification

## Abstract

Concept bottleneck models (CBMs) provide an interpretable framework for medical image classification by predicting human-understandable concepts before diagnosis. However, existing CBMs often rely on weakly grounded concept discovery and model concepts as isolated, deterministic variables, limiting their ability to capture clinically meaningful concepts, diagnostic uncertainty, and higher-order interactions. To address these limitations, we propose SHAPE, the State-Aware HypergrAph concePt bottlEneck model, for interpretable medical image classification. SHAPE models concepts from three complementary perspectives: discovery, uncertainty modeling, and structured reasoning. Specifically, DACD exploits representative and boundary-sensitive exemplars to generate grounded medical concepts. SACM represents each concept as a distribution over diagnostic states, enabling uncertainty-aware concept modeling at the concept level rather than prediction level. DCR captures class-conditional diagnostic cue structures via hypergraph reasoning, enabling predictions guided by structured concept combinations instead of independent concepts. Extensive experiments on six medical imaging datasets across nine classification settings show that SHAPE achieves competitive or superior performance over recent CBM baselines while providing more clinically meaningful explanations. These results demonstrate the effectiveness of concept-level uncertainty modeling and structured diagnostic reasoning for trustworthy medical image interpretation.

<p align="center">
  <img src="main figure.png" width="900">
</p>

## Included modules

- `run_train.py`: main training entry
- `scripts/discover_concepts.py`: concept discovery for **Distribution-Aware Concept Discovery (DACD)**
- `utils/exemplar_selection.py`: representative and boundary exemplar selection for DACD
- `utils/diagnostic_evidence_distribution.py`: concept-state distribution modeling for **State-Aware Concept Modeling (SACM)**
- `utils/structured_diagnostic_cues.py`: pairwise cue construction used in **Diagnostic Cue Reasoning (DCR)**
- `utils/hypergraph_diagnostic_cues.py`: hypergraph-based higher-order reasoning used in **Diagnostic Cue Reasoning (DCR)**
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
- plotting and analysis utilities
- checkpoints and saved outputs
- temporary experiment files

## Core pipeline

SHAPE contains three main components:

1. **Distribution-Aware Concept Discovery (DACD)**  
   Generates dataset-adaptive medical concepts from representative and boundary-sensitive exemplars.

2. **State-Aware Concept Modeling (SACM)**  
   Represents each concept as a distribution over diagnostic states to model concept strength and uncertainty.

3. **Diagnostic Cue Reasoning (DCR)**  
   Performs structured diagnostic reasoning over concept relations, including pairwise cue construction and hypergraph-based higher-order cue modeling.
