import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class DiagnosticCueStructure:
    class_index: int
    class_name: str
    concept_indices: List[int]
    concept_names: List[str]
    score: float
    positive_coactivation: float
    negative_coactivation: float

    @property
    def name(self) -> str:
        return " + ".join(self.concept_names)

    def to_dict(self) -> Dict:
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "concept_indices": self.concept_indices,
            "concept_names": self.concept_names,
            "cue_name": self.name,
            "score": self.score,
            "positive_coactivation": self.positive_coactivation,
            "negative_coactivation": self.negative_coactivation,
        }


class StructuredDiagnosticCueModule:
    """Mine class-conditional higher-order cue structures from concept logits."""

    def __init__(
        self,
        top_cue_structures_per_class: int = 5,
        cue_group_size: int = 2,
        cue_binarize_threshold: float = 0.5,
        cue_activation_type: str = "product",
    ):
        if cue_group_size != 2:
            raise ValueError("The first SDCM version supports only pair-level cue_group_size=2.")
        if cue_activation_type not in {"product", "min", "mean"}:
            raise ValueError(
                "cue_activation_type must be one of: product, min, mean."
            )

        self.top_cue_structures_per_class = top_cue_structures_per_class
        self.cue_group_size = cue_group_size
        self.cue_binarize_threshold = cue_binarize_threshold
        self.cue_activation_type = cue_activation_type
        self.cue_structures: List[DiagnosticCueStructure] = []

    def fit(
        self,
        concept_logits: torch.Tensor,
        labels: torch.Tensor,
        concept_names: Sequence[str],
        class_names: Sequence[str],
    ) -> List[DiagnosticCueStructure]:
        concept_probs = torch.sigmoid(concept_logits.detach().cpu())
        binary_concepts = concept_probs >= self.cue_binarize_threshold
        labels = labels.detach().cpu().long()

        n_concepts = binary_concepts.shape[1]
        pairs = list(itertools.combinations(range(n_concepts), self.cue_group_size))
        pair_tensor = torch.tensor(pairs, dtype=torch.long)
        pair_active = (
            binary_concepts[:, pair_tensor[:, 0]]
            & binary_concepts[:, pair_tensor[:, 1]]
        ).float()

        cue_structures: List[DiagnosticCueStructure] = []
        for class_idx, class_name in enumerate(class_names):
            positive_mask = labels == class_idx
            negative_mask = labels != class_idx

            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue

            positive_rate = pair_active[positive_mask].mean(dim=0)
            negative_rate = pair_active[negative_mask].mean(dim=0)
            scores = positive_rate - negative_rate

            k = min(self.top_cue_structures_per_class, len(pairs))
            top_scores, top_indices = torch.topk(scores, k=k)

            for score, pair_idx in zip(top_scores.tolist(), top_indices.tolist()):
                pair = list(pairs[pair_idx])
                cue_structures.append(
                    DiagnosticCueStructure(
                        class_index=class_idx,
                        class_name=class_name,
                        concept_indices=pair,
                        concept_names=[concept_names[i] for i in pair],
                        score=float(score),
                        positive_coactivation=float(positive_rate[pair_idx].item()),
                        negative_coactivation=float(negative_rate[pair_idx].item()),
                    )
                )

        self.cue_structures = cue_structures
        return cue_structures

    def transform(self, concept_logits: torch.Tensor) -> torch.Tensor:
        if not self.cue_structures:
            return concept_logits.new_zeros((concept_logits.shape[0], 0))

        concept_probs = torch.sigmoid(concept_logits)
        cue_values = []
        for cue in self.cue_structures:
            group = concept_probs[:, cue.concept_indices]
            if self.cue_activation_type == "product":
                cue_value = group.prod(dim=1)
            elif self.cue_activation_type == "min":
                cue_value = group.min(dim=1).values
            elif self.cue_activation_type == "mean":
                cue_value = group.mean(dim=1)
            else:
                raise ValueError(f"Unsupported cue activation type: {self.cue_activation_type}")
            cue_values.append(cue_value)

        return torch.stack(cue_values, dim=1)

    def fit_transform(
        self,
        concept_logits: torch.Tensor,
        labels: torch.Tensor,
        concept_names: Sequence[str],
        class_names: Sequence[str],
    ) -> torch.Tensor:
        self.fit(concept_logits, labels, concept_names, class_names)
        return self.transform(concept_logits)

    def state_dict(self) -> Dict:
        return {
            "top_cue_structures_per_class": self.top_cue_structures_per_class,
            "cue_group_size": self.cue_group_size,
            "cue_binarize_threshold": self.cue_binarize_threshold,
            "cue_activation_type": self.cue_activation_type,
            "cue_structures": [cue.to_dict() for cue in self.cue_structures],
        }

    def save(self, save_dir: str) -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "diagnostic_cue_structures.json")
        csv_path = os.path.join(save_dir, "diagnostic_cue_structures.csv")

        payload = self.state_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_index",
                    "class_name",
                    "cue_name",
                    "concept_indices",
                    "concept_names",
                    "score",
                    "positive_coactivation",
                    "negative_coactivation",
                ],
            )
            writer.writeheader()
            for cue in self.cue_structures:
                row = cue.to_dict()
                row["concept_indices"] = "|".join(str(i) for i in row["concept_indices"])
                row["concept_names"] = "|".join(row["concept_names"])
                writer.writerow(row)

        return {"json": json_path, "csv": csv_path}
