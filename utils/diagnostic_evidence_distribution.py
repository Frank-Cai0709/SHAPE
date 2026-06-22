import csv
import json
import os
from typing import Dict, Sequence

import torch
import torch.nn.functional as F


class DiagnosticEvidenceDistributionModule:
    """Convert concept head outputs into evidential diagnostic state distributions."""

    def __init__(
        self,
        num_concept_states: int = 4,
        concept_state_names: Sequence[str] = ("absent", "weak", "moderate", "strong"),
        uncertainty_weight: float = 0.01,
        kl_weight: float = 0.0,
    ):
        if num_concept_states < 2:
            raise ValueError("num_concept_states must be at least 2.")
        if len(concept_state_names) != num_concept_states:
            raise ValueError(
                "concept_state_names length must match num_concept_states."
            )

        self.num_concept_states = num_concept_states
        self.concept_state_names = list(concept_state_names)
        self.uncertainty_weight = uncertainty_weight
        self.kl_weight = kl_weight
        self.strength_weights = torch.linspace(0.0, 1.0, num_concept_states)

    def dirichlet_kl_to_uniform(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL(Dir(alpha) || Dir(1)) for each concept distribution."""
        alpha = alpha.clamp_min(1e-8)
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        num_states = alpha.shape[-1]

        log_norm_ratio = torch.lgamma(alpha_sum).squeeze(-1) - torch.lgamma(alpha).sum(dim=-1)
        log_uniform_norm = torch.lgamma(
            torch.tensor(float(num_states), device=alpha.device, dtype=alpha.dtype)
        )
        digamma_term = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha_sum))).sum(dim=-1)
        return log_norm_ratio - log_uniform_norm + digamma_term

    def forward(self, raw_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(raw_output)
        alpha = evidence + 1.0
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        prob = alpha / alpha_sum

        strength_weights = self.strength_weights.to(raw_output.device, raw_output.dtype)
        strength = (prob * strength_weights).sum(dim=-1)
        uncertainty = self.num_concept_states / alpha_sum.squeeze(-1)
        reliable_concept = strength * (1.0 - uncertainty)
        most_likely_state = prob.argmax(dim=-1)

        return {
            "raw_output": raw_output,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "strength": strength,
            "uncertainty": uncertainty,
            "reliable_concept": reliable_concept,
            "most_likely_state": most_likely_state,
        }

    def binary_labels_to_state_targets(self, binary_labels: torch.Tensor) -> torch.Tensor:
        state_targets = torch.zeros_like(binary_labels, dtype=torch.long)
        state_targets[binary_labels > 0.5] = self.num_concept_states - 1
        return state_targets

    def concept_labels_to_state_distribution(self, concept_labels: torch.Tensor) -> torch.Tensor:
        """Map hard or soft concept labels to a soft evidential state target."""
        labels = concept_labels.to(dtype=torch.float32).clamp(0.0, 1.0)
        target = torch.zeros(
            *labels.shape,
            self.num_concept_states,
            dtype=labels.dtype,
            device=labels.device,
        )
        target[..., 0] = 1.0 - labels
        target[..., self.num_concept_states - 1] = labels
        return target

    def loss(self, raw_output: torch.Tensor, binary_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.forward(raw_output)
        state_targets = self.concept_labels_to_state_distribution(binary_labels).to(raw_output.device)
        log_prob = torch.log(outputs["prob"].clamp_min(1e-8))
        state_loss = -(state_targets * log_prob).sum(dim=-1).mean()
        uncertainty_loss = outputs["uncertainty"].mean()
        kl_per_concept = self.dirichlet_kl_to_uniform(outputs["alpha"])
        gated_kl_loss = (outputs["uncertainty"].detach() * kl_per_concept).mean()
        total_loss = (
            state_loss
            + self.uncertainty_weight * uncertainty_loss
            + self.kl_weight * gated_kl_loss
        )

        return {
            "loss": total_loss,
            "state_loss": state_loss,
            "uncertainty_loss": uncertainty_loss,
            "gated_kl_loss": gated_kl_loss,
            "outputs": outputs,
        }

    def state_dict(self) -> Dict:
        return {
            "num_concept_states": self.num_concept_states,
            "concept_state_names": self.concept_state_names,
            "uncertainty_weight": self.uncertainty_weight,
            "kl_weight": self.kl_weight,
            "strength_weights": self.strength_weights.tolist(),
        }

    def summarize_uncertainty(self, uncertainty: torch.Tensor) -> Dict[str, float]:
        values = uncertainty.detach().cpu().flatten()
        return {
            "mean": float(values.mean().item()),
            "std": float(values.std().item()),
            "min": float(values.min().item()),
            "q25": float(values.quantile(0.25).item()),
            "median": float(values.median().item()),
            "q75": float(values.quantile(0.75).item()),
            "max": float(values.max().item()),
        }

    def save_outputs(
        self,
        save_dir: str,
        split_outputs: Dict[str, Dict[str, torch.Tensor]],
        split_labels: Dict[str, torch.Tensor],
        concept_names: Sequence[str],
        class_names: Sequence[str],
    ) -> Dict[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        sample_csv_path = os.path.join(save_dir, "diagnostic_evidence_samples.csv")
        class_json_path = os.path.join(save_dir, "diagnostic_evidence_class_summary.json")
        class_csv_path = os.path.join(save_dir, "diagnostic_evidence_class_summary.csv")

        class_summary = {}
        with open(sample_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "sample_index",
                    "label_index",
                    "class_name",
                    "concept_index",
                    "concept_name",
                    "strength",
                    "uncertainty",
                    "reliable_concept",
                    "most_likely_state",
                ],
            )
            writer.writeheader()

            for split, outputs in split_outputs.items():
                labels = split_labels[split].detach().cpu().long()
                strength = outputs["strength"].detach().cpu()
                uncertainty = outputs["uncertainty"].detach().cpu()
                reliable = outputs["reliable_concept"].detach().cpu()
                state_idx = outputs["most_likely_state"].detach().cpu()

                class_summary[split] = {}
                for class_idx, class_name in enumerate(class_names):
                    mask = labels == class_idx
                    if mask.sum() == 0:
                        continue

                    class_summary[split][class_name] = {
                        "num_samples": int(mask.sum().item()),
                        "mean_strength": strength[mask].mean(dim=0).tolist(),
                        "mean_uncertainty": uncertainty[mask].mean(dim=0).tolist(),
                        "mean_reliable_concept": reliable[mask].mean(dim=0).tolist(),
                        "concept_names": list(concept_names),
                    }

                for sample_idx in range(strength.shape[0]):
                    label_idx = int(labels[sample_idx].item())
                    class_name = class_names[label_idx]
                    for concept_idx, concept_name in enumerate(concept_names):
                        writer.writerow(
                            {
                                "split": split,
                                "sample_index": sample_idx,
                                "label_index": label_idx,
                                "class_name": class_name,
                                "concept_index": concept_idx,
                                "concept_name": concept_name,
                                "strength": float(strength[sample_idx, concept_idx].item()),
                                "uncertainty": float(uncertainty[sample_idx, concept_idx].item()),
                                "reliable_concept": float(reliable[sample_idx, concept_idx].item()),
                                "most_likely_state": self.concept_state_names[
                                    int(state_idx[sample_idx, concept_idx].item())
                                ],
                            }
                        )

        with open(class_json_path, "w", encoding="utf-8") as f:
            json.dump(class_summary, f, indent=2, ensure_ascii=False)

        with open(class_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "class_name",
                    "concept_index",
                    "concept_name",
                    "mean_strength",
                    "mean_uncertainty",
                    "mean_reliable_concept",
                ],
            )
            writer.writeheader()
            for split, split_summary in class_summary.items():
                for class_name, summary in split_summary.items():
                    for concept_idx, concept_name in enumerate(concept_names):
                        writer.writerow(
                            {
                                "split": split,
                                "class_name": class_name,
                                "concept_index": concept_idx,
                                "concept_name": concept_name,
                                "mean_strength": summary["mean_strength"][concept_idx],
                                "mean_uncertainty": summary["mean_uncertainty"][concept_idx],
                                "mean_reliable_concept": summary["mean_reliable_concept"][concept_idx],
                            }
                        )

        return {
            "sample_csv": sample_csv_path,
            "class_json": class_json_path,
            "class_csv": class_csv_path,
        }
