import csv
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class Hyperedge:
    class_index: int
    class_name: str
    concept_indices: List[int]
    concept_names: List[str]
    score: float
    discriminativeness: float
    synergy: float
    positive_frequency: float
    negative_frequency: float
    expected_positive_frequency: float
    source: str = "hdcm"

    @property
    def size(self) -> int:
        return len(self.concept_indices)

    @property
    def name(self) -> str:
        return " + ".join(self.concept_names)

    def to_dict(self) -> Dict:
        return {
            "class_index": self.class_index,
            "class_name": self.class_name,
            "concept_indices": self.concept_indices,
            "concept_names": self.concept_names,
            "hyperedge_name": self.name,
            "hyperedge_size": self.size,
            "score": self.score,
            "discriminativeness": self.discriminativeness,
            "synergy": self.synergy,
            "positive_frequency": self.positive_frequency,
            "negative_frequency": self.negative_frequency,
            "expected_positive_frequency": self.expected_positive_frequency,
            "source": self.source,
        }


class HypergraphDiagnosticCueModule:
    """Mine and activate hypergraph-based higher-order diagnostic cue structures."""

    def __init__(
        self,
        hyperedge_size: int = 3,
        top_hyperedges_per_class: int = 5,
        hyperedge_activation_type: str = "product",
        hyperedge_binarize_threshold: float = 0.5,
        hyperedge_synergy_weight: float = 1.0,
        max_candidate_concepts_per_class: int = 30,
        use_hypergraph_message_passing: bool = False,
    ):
        if hyperedge_size not in {2, 3, 4}:
            raise ValueError("DCR supports hyperedge_size 2, 3, or 4.")
        if hyperedge_activation_type not in {"product", "min", "mean", "noisy_and"}:
            raise ValueError(
                "hyperedge_activation_type must be one of: product, min, mean, noisy_and."
            )

        self.hyperedge_size = hyperedge_size
        self.top_hyperedges_per_class = top_hyperedges_per_class
        self.hyperedge_activation_type = hyperedge_activation_type
        self.hyperedge_binarize_threshold = hyperedge_binarize_threshold
        self.hyperedge_synergy_weight = hyperedge_synergy_weight
        self.max_candidate_concepts_per_class = max_candidate_concepts_per_class
        self.use_hypergraph_message_passing = use_hypergraph_message_passing
        self.hyperedges: List[Hyperedge] = []
        self.incidence_matrix: Optional[torch.Tensor] = None

    def _score_edge(
        self,
        binary_concepts: torch.Tensor,
        labels: torch.Tensor,
        class_index: int,
        indices: Sequence[int],
    ) -> Dict[str, float]:
        positive_mask = labels == class_index
        negative_mask = labels != class_index
        active = binary_concepts[:, list(indices)].all(dim=1).float()

        positive_frequency = active[positive_mask].mean().item() if positive_mask.any() else 0.0
        negative_frequency = active[negative_mask].mean().item() if negative_mask.any() else 0.0
        discriminativeness = positive_frequency - negative_frequency

        if positive_mask.any():
            member_frequencies = binary_concepts[positive_mask][:, list(indices)].float().mean(dim=0)
            expected_positive_frequency = member_frequencies.prod().item()
        else:
            expected_positive_frequency = 0.0

        synergy = positive_frequency - expected_positive_frequency
        score = discriminativeness + self.hyperedge_synergy_weight * synergy

        return {
            "score": score,
            "discriminativeness": discriminativeness,
            "synergy": synergy,
            "positive_frequency": positive_frequency,
            "negative_frequency": negative_frequency,
            "expected_positive_frequency": expected_positive_frequency,
        }

    def _to_probabilities(self, concept_activations: torch.Tensor) -> torch.Tensor:
        values = concept_activations.detach().cpu().float()
        if values.numel() == 0:
            return values
        if values.min() < 0.0 or values.max() > 1.0:
            return torch.sigmoid(values)
        return values.clamp(0.0, 1.0)

    def _candidate_edges_for_class(
        self,
        binary_concepts: torch.Tensor,
        class_mask: torch.Tensor,
    ) -> List[tuple]:
        class_active = binary_concepts[class_mask].float()
        if class_active.numel() == 0:
            return []

        frequencies = class_active.mean(dim=0)
        k = min(self.max_candidate_concepts_per_class, frequencies.shape[0])
        top_concepts = torch.topk(frequencies, k=k).indices.tolist()
        return list(itertools.combinations(top_concepts, self.hyperedge_size))

    def fit(
        self,
        concept_activations: torch.Tensor,
        labels: torch.Tensor,
        concept_names: Sequence[str],
        class_names: Sequence[str],
        initial_hyperedges: Optional[Sequence[Dict]] = None,
    ) -> List[Hyperedge]:
        concept_probs = self._to_probabilities(concept_activations)
        binary_concepts = concept_probs >= self.hyperedge_binarize_threshold
        labels = labels.detach().cpu().long()

        hyperedges: List[Hyperedge] = []
        seen = set()

        if initial_hyperedges:
            for edge in initial_hyperedges:
                indices = tuple(edge["concept_indices"])
                class_index = int(edge["class_index"])
                key = (class_index, indices)
                if key in seen:
                    continue
                seen.add(key)
                edge_scores = self._score_edge(
                    binary_concepts, labels, class_index, indices
                )
                hyperedges.append(
                    Hyperedge(
                        class_index=class_index,
                        class_name=class_names[class_index],
                        concept_indices=list(indices),
                        concept_names=[concept_names[i] for i in indices],
                        score=edge_scores["score"],
                        discriminativeness=edge_scores["discriminativeness"],
                        synergy=edge_scores["synergy"],
                        positive_frequency=edge_scores["positive_frequency"],
                        negative_frequency=edge_scores["negative_frequency"],
                        expected_positive_frequency=edge_scores["expected_positive_frequency"],
                        source=edge.get("source", "sdcm_seed"),
                    )
                )

        for class_index, class_name in enumerate(class_names):
            positive_mask = labels == class_index
            negative_mask = labels != class_index
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue

            candidates = self._candidate_edges_for_class(binary_concepts, positive_mask)
            scored_edges = []
            for candidate in candidates:
                key = (class_index, tuple(candidate))
                if key in seen:
                    continue
                edge_scores = self._score_edge(
                    binary_concepts, labels, class_index, candidate
                )
                score = edge_scores["score"]
                scored_edges.append(
                    (
                        score,
                        Hyperedge(
                            class_index=class_index,
                            class_name=class_name,
                            concept_indices=list(candidate),
                            concept_names=[concept_names[i] for i in candidate],
                            score=score,
                            discriminativeness=edge_scores["discriminativeness"],
                            synergy=edge_scores["synergy"],
                            positive_frequency=edge_scores["positive_frequency"],
                            negative_frequency=edge_scores["negative_frequency"],
                            expected_positive_frequency=edge_scores["expected_positive_frequency"],
                            source="hdcm",
                        ),
                    )
                )

            scored_edges.sort(key=lambda item: item[0], reverse=True)
            for _, edge in scored_edges[: self.top_hyperedges_per_class]:
                seen.add((edge.class_index, tuple(edge.concept_indices)))
                hyperedges.append(edge)

        self.hyperedges = hyperedges
        self.incidence_matrix = self.build_incidence_matrix(len(concept_names))
        return hyperedges

    def build_incidence_matrix(self, num_concepts: int) -> torch.Tensor:
        incidence = torch.zeros(num_concepts, len(self.hyperedges))
        for edge_idx, edge in enumerate(self.hyperedges):
            incidence[edge.concept_indices, edge_idx] = 1.0
        return incidence

    def transform(self, concept_activations: torch.Tensor) -> torch.Tensor:
        if not self.hyperedges:
            return concept_activations.new_zeros((concept_activations.shape[0], 0))

        concept_probs = self._to_probabilities(concept_activations).to(
            concept_activations.device
        )
        hyperedge_values = []
        for edge in self.hyperedges:
            group = concept_probs[:, edge.concept_indices]
            if self.hyperedge_activation_type == "product":
                value = group.prod(dim=1)
            elif self.hyperedge_activation_type == "min":
                value = group.min(dim=1).values
            elif self.hyperedge_activation_type == "mean":
                value = group.mean(dim=1)
            elif self.hyperedge_activation_type == "noisy_and":
                value = torch.sigmoid(10.0 * (group.mean(dim=1) - 0.5))
            else:
                raise ValueError(
                    f"Unsupported hyperedge activation type: {self.hyperedge_activation_type}"
                )
            hyperedge_values.append(value)
        return torch.stack(hyperedge_values, dim=1)

    def fit_transform(
        self,
        concept_activations: torch.Tensor,
        labels: torch.Tensor,
        concept_names: Sequence[str],
        class_names: Sequence[str],
        initial_hyperedges: Optional[Sequence[Dict]] = None,
    ) -> torch.Tensor:
        self.fit(concept_activations, labels, concept_names, class_names, initial_hyperedges)
        return self.transform(concept_activations)

    def propagate(self, concept_features: torch.Tensor) -> torch.Tensor:
        if self.incidence_matrix is None or self.incidence_matrix.numel() == 0:
            return concept_features.new_zeros(concept_features.shape)

        h = self.incidence_matrix.to(concept_features.device, concept_features.dtype)
        dv = h.sum(dim=1).clamp_min(1e-6)
        de = h.sum(dim=0).clamp_min(1e-6)
        dv_inv_sqrt = torch.diag(torch.pow(dv, -0.5))
        de_inv = torch.diag(torch.pow(de, -1.0))
        propagation = dv_inv_sqrt @ h @ de_inv @ h.T @ dv_inv_sqrt
        return concept_features @ propagation.T

    def enhanced_features(self, concept_features: torch.Tensor) -> torch.Tensor:
        hyperedge_activations = self.transform(concept_features)
        features = [concept_features, hyperedge_activations]
        if self.use_hypergraph_message_passing:
            features.append(self.propagate(concept_features))
        return torch.cat(features, dim=1)

    def state_dict(self) -> Dict:
        return {
            "hyperedge_size": self.hyperedge_size,
            "top_hyperedges_per_class": self.top_hyperedges_per_class,
            "hyperedge_activation_type": self.hyperedge_activation_type,
            "hyperedge_binarize_threshold": self.hyperedge_binarize_threshold,
            "hyperedge_synergy_weight": self.hyperedge_synergy_weight,
            "use_hypergraph_message_passing": self.use_hypergraph_message_passing,
            "hyperedges": [edge.to_dict() for edge in self.hyperedges],
            "incidence_matrix": None
            if self.incidence_matrix is None
            else self.incidence_matrix.tolist(),
        }

    def save(
        self,
        save_dir: str,
        split_activations: Optional[Dict[str, torch.Tensor]] = None,
        top_k_per_sample: int = 5,
    ) -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, "hypergraph_diagnostic_cue_structures.json")
        csv_path = os.path.join(save_dir, "hypergraph_diagnostic_cue_structures.csv")
        sample_csv_path = os.path.join(save_dir, "hypergraph_top_activated_hyperedges.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, indent=2, ensure_ascii=False)

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_index",
                    "class_name",
                    "hyperedge_name",
                    "hyperedge_size",
                    "concept_indices",
                    "concept_names",
                    "score",
                    "discriminativeness",
                    "synergy",
                    "positive_frequency",
                    "negative_frequency",
                    "expected_positive_frequency",
                    "source",
                ],
            )
            writer.writeheader()
            for edge in self.hyperedges:
                row = edge.to_dict()
                row["concept_indices"] = "|".join(str(i) for i in row["concept_indices"])
                row["concept_names"] = "|".join(row["concept_names"])
                writer.writerow(row)

        if split_activations:
            with open(sample_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "split",
                        "sample_index",
                        "rank",
                        "hyperedge_index",
                        "hyperedge_name",
                        "class_name",
                        "activation",
                        "source",
                    ],
                )
                writer.writeheader()
                for split, activations in split_activations.items():
                    if activations.shape[1] == 0:
                        continue
                    k = min(top_k_per_sample, activations.shape[1])
                    top_values, top_indices = torch.topk(activations.detach().cpu(), k=k, dim=1)
                    for sample_idx in range(top_indices.shape[0]):
                        for rank in range(k):
                            edge_idx = int(top_indices[sample_idx, rank].item())
                            edge = self.hyperedges[edge_idx]
                            writer.writerow(
                                {
                                    "split": split,
                                    "sample_index": sample_idx,
                                    "rank": rank + 1,
                                    "hyperedge_index": edge_idx,
                                    "hyperedge_name": edge.name,
                                    "class_name": edge.class_name,
                                    "activation": float(top_values[sample_idx, rank].item()),
                                    "source": edge.source,
                                }
                            )

        return {
            "json": json_path,
            "csv": csv_path,
            "sample_top_csv": sample_csv_path,
        }
