import torch
import os
import random
import utils_my
import data_utils
import argparse
import datetime
import json
import csv
from loguru import logger
import sys
import numpy as np
from collections import defaultdict
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from model.cbl import ConceptBottleneckLayer
from utils.diagnostic_evidence_distribution import DiagnosticEvidenceDistributionModule
from utils.hypergraph_diagnostic_cues import HypergraphDiagnosticCueModule
from utils.structured_diagnostic_cues import StructuredDiagnosticCueModule

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def train_test_cbm_and_save(args):
    # Setup log directory and logger
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = f"{args.save_dir}/{args.dataset}/{args.dataset}_cbm_{timestamp}"
    while os.path.exists(save_dir):
        save_dir += "-1"
    os.makedirs(save_dir)
    
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info(f"Saving model to {save_dir}")
    
    # Load classes and concepts
    classes = data_utils.get_class_names(args.dataset)
    
    concept_set = args.concept_set or f"Concept generate/{args.dataset}_concept_class_wimage.json"
    with open(concept_set, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    concepts = concepts_data["concepts"]
    concepts_to_class = concepts_data["concepts_to_class"]
    concept_state_names = [name.strip() for name in args.concept_state_names.split(",") if name.strip()]

    # Prepare datasets
    d_train = f"{args.dataset}_train"
    d_val = f"{args.dataset}_val"
    d_test = f"{args.dataset}_test"
    
    # Get image-to-class mappings
    images_to_class_train = data_utils.get_targets_only(*d_train.split('_'))
    images_to_class_val = data_utils.get_targets_only(*d_val.split('_'))
    images_to_class_test = data_utils.get_targets_only(*d_test.split('_'))

    # Save activations for all datasets
    for d_probe in [d_train, d_val, d_test]:
        utils_my.save_all_features(
            clip_name=args.clip_name,
            target_name=args.backbone,
            target_layers=[args.feature_layer],
            d_probe=d_probe,
            concept_set_path=concept_set,
            batch_size=args.batch_size,
            device=args.device,
            pool_mode="avg",
            save_dir=args.activation_dir,
            use_penultimate=args.use_penultimate
        )

    # Load features with consistent naming
    def load_features(d_name):
        paths = utils_my.get_feature_paths(
            args.clip_name, 
            args.backbone, 
            d_name, 
            concept_set,
            [args.feature_layer] if args.feature_layer else None, 
            "avg", 
            args.activation_dir,
            use_penultimate=args.use_penultimate
        )
        
        with torch.no_grad():

            if isinstance(paths["target"], dict): 
                target_feats = torch.load(paths["target"][args.feature_layer], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###
            else: 
                target_feats = torch.load(paths["target"], map_location="cpu").float()
                #target_feats /= target_feats.norm(dim=1, keepdim=True)###
                
            image_feats = torch.load(paths["clip"], map_location="cpu").float()
            image_feats /= image_feats.norm(dim=1, keepdim=True)
            text_feats = torch.load(paths["text"], map_location="cpu").float()
            text_feats /= text_feats.norm(dim=1, keepdim=True)
            clip_feats = image_feats @ text_feats.T
            
        return target_feats, image_feats, text_feats, clip_feats

    # Load all features
    train_target, train_image, train_text, train_clip = load_features(d_train)
    val_target, val_image, val_text, val_clip = load_features(d_val)
    test_target, test_image, _, _ = load_features(d_test)
    
    # Prepare targets (no splitting needed)
    train_targets = torch.LongTensor(images_to_class_train)
    val_targets = torch.LongTensor(images_to_class_val)
    test_targets = torch.LongTensor(images_to_class_test)

    max_val = torch.max(train_clip).item() 
    min_val = torch.min(train_clip).item()
    mean_val = torch.mean(train_clip).item() 
    
    print(f"Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}")

    # Filter and merge concepts (using CLIP features from train set)
    (new_concepts, 
     new_text_features, 
     concept_to_classes, 
     concept_redirect_map) = utils_my.filter_and_merge_concepts(
        train_clip, train_text, concepts, concepts_to_class,
        images_to_class_train,  # Use full train set
        Tconf=args.Tconf, Tmerge=args.Tmerge, K=4, strategy="max",K_indep=args.K_indep
    )

    # Generate concept pseudo-labels. Adaptive thresholds are calibrated on train
    # only, then reused for val/test to avoid split-specific leakage.
    concept_labels_train, concept_label_info_train = utils_my.generate_concept_labels(
        train_image,
        new_text_features,
        images_to_class_train,
        concept_to_classes,
        Tconf=args.Tconf,
        use_soft_concept_labels=args.use_soft_concept_labels,
        adaptive_threshold=args.adaptive_threshold,
        lambda_std=args.lambda_std,
        temperature=args.concept_label_temperature,
        return_info=True,
    )
    concept_threshold_stats = concept_label_info_train["threshold_stats"]
    concept_labels_val, concept_label_info_val = utils_my.generate_concept_labels(
        val_image,
        new_text_features,
        images_to_class_val,
        concept_to_classes,
        Tconf=args.Tconf,
        use_soft_concept_labels=args.use_soft_concept_labels,
        adaptive_threshold=args.adaptive_threshold,
        lambda_std=args.lambda_std,
        temperature=args.concept_label_temperature,
        threshold_stats=concept_threshold_stats,
        return_info=True,
    )
    concept_labels_test, concept_label_info_test = utils_my.generate_concept_labels(
        test_image,
        new_text_features,
        images_to_class_test,
        concept_to_classes,
        Tconf=args.Tconf,
        use_soft_concept_labels=args.use_soft_concept_labels,
        adaptive_threshold=args.adaptive_threshold,
        lambda_std=args.lambda_std,
        temperature=args.concept_label_temperature,
        threshold_stats=concept_threshold_stats,
        return_info=True,
    )
    concept_binary_labels_train = concept_label_info_train["binary_labels"]
    concept_binary_labels_val = concept_label_info_val["binary_labels"]
    concept_binary_labels_test = concept_label_info_test["binary_labels"]
    logger.info(f"Soft concept labels enabled: {args.use_soft_concept_labels}")
    logger.info(f"Adaptive concept thresholds enabled: {args.adaptive_threshold}")
    logger.info(f"Concept label temperature: {args.concept_label_temperature:.4f}")
    logger.info(f"Adaptive threshold lambda_std: {args.lambda_std:.4f}")


    # Train Concept Bottleneck Layer (using full train set)
    dedm = None
    cbl_concept_dim = len(new_concepts)
    if args.use_dedm:
        dedm = DiagnosticEvidenceDistributionModule(
            num_concept_states=args.num_concept_states,
            concept_state_names=concept_state_names,
            uncertainty_weight=args.dedm_uncertainty_weight,
            kl_weight=args.dedm_kl_weight,
        )
        cbl_concept_dim = len(new_concepts) * args.num_concept_states
        logger.info("DEDM enabled: True")
        logger.info(f"Concept count: {len(new_concepts)}")
        logger.info(f"Concept state count: {args.num_concept_states}")
        logger.info(f"Concept state names: {concept_state_names}")
        logger.info(f"DEDM uncertainty weight: {args.dedm_uncertainty_weight}")
        logger.info(f"DEDM uncertainty-gated KL weight: {args.dedm_kl_weight}")
    else:
        logger.info("DEDM enabled: False")
        logger.info(f"Concept count: {len(new_concepts)}")

    cbl = ConceptBottleneckLayer(
        input_dim=train_target.shape[1],
        concept_dim=cbl_concept_dim,
        cbl_layer_num=args.cbl_layer_num,
        bias=args.cbl_bias
    ).to(args.device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(cbl.parameters(), lr=args.cbl_lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    best_weights = None
    
    for step in range(args.cbl_steps):
        # Mini-batch training (full train set)
        idx = torch.randperm(len(train_target))[:args.cbl_batch_size]
        feats = train_target[idx].to(args.device)
        labels = concept_labels_train[idx].to(args.device)
        
        optimizer.zero_grad()
        outputs = cbl(feats)
        if args.use_dedm:
            outputs = outputs.view(-1, len(new_concepts), args.num_concept_states)
            loss_parts = dedm.loss(outputs, labels)
            loss = loss_parts["loss"]
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Validation (full val set)
        if step % 50 == 0 or step == args.cbl_steps - 1:
            with torch.no_grad():
                val_outputs = cbl(val_target.to(args.device))
                if args.use_dedm:
                    val_outputs = val_outputs.view(-1, len(new_concepts), args.num_concept_states)
                    val_loss_parts = dedm.loss(val_outputs, concept_labels_val.to(args.device))
                    val_loss = val_loss_parts["loss"]
                else:
                    val_loss = criterion(val_outputs, concept_labels_val.to(args.device))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = cbl.state_dict()
                    if args.use_dedm:
                        logger.info(
                            f"Step {step}: Train Loss {loss.item():.4f}, "
                            f"State Loss {loss_parts['state_loss'].item():.4f}, "
                            f"Uncertainty {loss_parts['uncertainty_loss'].item():.4f}, "
                            f"Gated KL {loss_parts['gated_kl_loss'].item():.4f}, "
                            f"Val Loss {val_loss.item():.4f}"
                        )
                    else:
                        logger.info(f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}")

    # Load best weights
    cbl.load_state_dict(best_weights)
    
    dedm_state = None
    dedm_paths = {}
    dedm_uncertainty_summary = {}

    # Extract and normalize features
    with torch.no_grad():
        if args.use_dedm:
            train_raw = cbl(train_target.to(args.device)).view(-1, len(new_concepts), args.num_concept_states)
            val_raw = cbl(val_target.to(args.device)).view(-1, len(new_concepts), args.num_concept_states)
            test_raw = cbl(test_target.to(args.device)).view(-1, len(new_concepts), args.num_concept_states)

            train_dedm = dedm.forward(train_raw)
            val_dedm = dedm.forward(val_raw)
            test_dedm = dedm.forward(test_raw)

            train_logits = train_dedm["reliable_concept"]
            val_logits = val_dedm["reliable_concept"]
            test_logits = test_dedm["reliable_concept"]

            dedm_state = dedm.state_dict()
            dedm_uncertainty_summary = {
                "train": dedm.summarize_uncertainty(train_dedm["uncertainty"]),
                "val": dedm.summarize_uncertainty(val_dedm["uncertainty"]),
                "test": dedm.summarize_uncertainty(test_dedm["uncertainty"]),
            }
            for split, summary in dedm_uncertainty_summary.items():
                logger.info(
                    f"DEDM uncertainty {split}: mean={summary['mean']:.4f}, "
                    f"median={summary['median']:.4f}, min={summary['min']:.4f}, max={summary['max']:.4f}"
                )
        else:
            train_dedm = None
            val_dedm = None
            test_dedm = None
            train_logits = cbl(train_target.to(args.device))
            val_logits = cbl(val_target.to(args.device))
            test_logits = cbl(test_target.to(args.device))

        train_mean = train_logits.mean(dim=0, keepdim=True)
        train_std = train_logits.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_c = (train_logits - train_mean) / train_std
        val_c = (val_logits - train_mean) / train_std
        test_c = (test_logits - train_mean) / train_std

        # Evaluation keeps concept predictions binary even when training uses soft labels.
        if args.use_dedm:
            concept_pred_train = (train_logits >= 0.5).float().cpu()
            concept_pred_val = (val_logits >= 0.5).float().cpu()
            concept_pred_test = (test_logits >= 0.5).float().cpu()
        else:
            concept_pred_train = (torch.sigmoid(train_logits) >= 0.5).float().cpu()
            concept_pred_val = (torch.sigmoid(val_logits) >= 0.5).float().cpu()
            concept_pred_test = (torch.sigmoid(test_logits) >= 0.5).float().cpu()

        concept_accuracy = {
            "train": (concept_pred_train == concept_binary_labels_train.cpu()).float().mean().item(),
            "val": (concept_pred_val == concept_binary_labels_val.cpu()).float().mean().item(),
            "test": (concept_pred_test == concept_binary_labels_test.cpu()).float().mean().item(),
        }

    sdcm_state = None
    cue_paths = {}
    hdcm_state = None
    hypergraph_paths = {}
    feature_names = list(new_concepts)
    logger.info(f"SDCM enabled: {args.use_sdcm}")
    logger.info(f"Feature dim before SDCM: {train_c.shape[1]}")
    if args.use_sdcm:
        sdcm = StructuredDiagnosticCueModule(
            top_cue_structures_per_class=args.top_cue_structures_per_class,
            cue_group_size=args.cue_group_size,
            cue_binarize_threshold=args.cue_binarize_threshold,
            cue_activation_type=args.cue_activation_type,
        )
        with torch.no_grad():
            if args.use_dedm:
                train_cue_source = torch.logit(train_logits.cpu().clamp(1e-6, 1 - 1e-6))
                val_cue_source = torch.logit(val_logits.cpu().clamp(1e-6, 1 - 1e-6))
                test_cue_source = torch.logit(test_logits.cpu().clamp(1e-6, 1 - 1e-6))
            else:
                train_cue_source = train_logits.cpu()
                val_cue_source = val_logits.cpu()
                test_cue_source = test_logits.cpu()

            train_cues = sdcm.fit_transform(
                train_cue_source, train_targets, new_concepts, classes
            )
            val_cues = sdcm.transform(val_cue_source)
            test_cues = sdcm.transform(test_cue_source)

            cue_mean = train_cues.mean(dim=0, keepdim=True)
            cue_std = train_cues.std(dim=0, keepdim=True).clamp_min(1e-6)
            train_cues = (train_cues - cue_mean) / cue_std
            val_cues = (val_cues - cue_mean) / cue_std
            test_cues = (test_cues - cue_mean) / cue_std

            train_c = torch.cat([train_c.cpu(), train_cues], dim=1)
            val_c = torch.cat([val_c.cpu(), val_cues], dim=1)
            test_c = torch.cat([test_c.cpu(), test_cues], dim=1)

        cue_paths = sdcm.save(save_dir)
        sdcm_state = sdcm.state_dict()
        sdcm_state["normalization"] = {
            "cue_mean": cue_mean.cpu(),
            "cue_std": cue_std.cpu(),
        }
        feature_names += [cue.name for cue in sdcm.cue_structures]

        class_to_count = defaultdict(int)
        for cue in sdcm.cue_structures:
            class_to_count[cue.class_name] += 1
        for class_name in classes:
            logger.info(
                f"SDCM cue structures for {class_name}: {class_to_count[class_name]}"
            )
        logger.info(f"SDCM cue structures json: {cue_paths['json']}")
        logger.info(f"SDCM cue structures csv: {cue_paths['csv']}")

    sdcm_feature_dim_after = train_c.shape[1]
    hdcm_feature_dim_before = train_c.shape[1]
    hdcm_hyperedge_feature_dim = 0
    logger.info(f"Feature dim after SDCM: {sdcm_feature_dim_after}")
    logger.info(f"DCR enabled: {args.use_hdcm}")
    logger.info(f"Feature dim before DCR: {hdcm_feature_dim_before}")
    if args.use_hdcm:
        hdcm = HypergraphDiagnosticCueModule(
            hyperedge_size=args.hyperedge_size,
            top_hyperedges_per_class=args.top_hyperedges_per_class,
            hyperedge_activation_type=args.hyperedge_activation_type,
            hyperedge_binarize_threshold=args.hyperedge_binarize_threshold,
            hyperedge_synergy_weight=args.hyperedge_synergy_weight,
            use_hypergraph_message_passing=args.use_hypergraph_message_passing,
        )

        sdcm_seed_hyperedges = []
        if sdcm_state is not None:
            for cue in sdcm_state["cue_structures"]:
                sdcm_seed_hyperedges.append(
                    {
                        "class_index": cue["class_index"],
                        "concept_indices": cue["concept_indices"],
                        "source": "sdcm_seed",
                    }
                )

        with torch.no_grad():
            train_hdcm_source = train_logits.cpu()
            val_hdcm_source = val_logits.cpu()
            test_hdcm_source = test_logits.cpu()

            train_hyperedges = hdcm.fit_transform(
                train_hdcm_source,
                train_targets,
                new_concepts,
                classes,
                initial_hyperedges=sdcm_seed_hyperedges,
            )
            val_hyperedges = hdcm.transform(val_hdcm_source)
            test_hyperedges = hdcm.transform(test_hdcm_source)

            hyperedge_mean = train_hyperedges.mean(dim=0, keepdim=True)
            hyperedge_std = train_hyperedges.std(dim=0, keepdim=True).clamp_min(1e-6)
            train_hyperedges_norm = (train_hyperedges - hyperedge_mean) / hyperedge_std
            val_hyperedges_norm = (val_hyperedges - hyperedge_mean) / hyperedge_std
            test_hyperedges_norm = (test_hyperedges - hyperedge_mean) / hyperedge_std

            train_hdcm_parts = [train_hyperedges_norm]
            val_hdcm_parts = [val_hyperedges_norm]
            test_hdcm_parts = [test_hyperedges_norm]

            hdcm_state = hdcm.state_dict()
            hdcm_state["normalization"] = {
                "hyperedge_mean": hyperedge_mean.cpu(),
                "hyperedge_std": hyperedge_std.cpu(),
            }

            if args.use_hypergraph_message_passing:
                train_propagated = hdcm.propagate(train_hdcm_source)
                val_propagated = hdcm.propagate(val_hdcm_source)
                test_propagated = hdcm.propagate(test_hdcm_source)
                propagated_mean = train_propagated.mean(dim=0, keepdim=True)
                propagated_std = train_propagated.std(dim=0, keepdim=True).clamp_min(1e-6)
                train_hdcm_parts.append((train_propagated - propagated_mean) / propagated_std)
                val_hdcm_parts.append((val_propagated - propagated_mean) / propagated_std)
                test_hdcm_parts.append((test_propagated - propagated_mean) / propagated_std)
                hdcm_state["normalization"]["propagated_mean"] = propagated_mean.cpu()
                hdcm_state["normalization"]["propagated_std"] = propagated_std.cpu()

            train_hdcm_features = torch.cat(train_hdcm_parts, dim=1)
            val_hdcm_features = torch.cat(val_hdcm_parts, dim=1)
            test_hdcm_features = torch.cat(test_hdcm_parts, dim=1)

            train_c = torch.cat([train_c.cpu(), train_hdcm_features], dim=1)
            val_c = torch.cat([val_c.cpu(), val_hdcm_features], dim=1)
            test_c = torch.cat([test_c.cpu(), test_hdcm_features], dim=1)
            hdcm_hyperedge_feature_dim = train_hyperedges.shape[1]

        feature_names += [edge.name for edge in hdcm.hyperedges]
        if args.use_hypergraph_message_passing:
            feature_names += [f"hypergraph propagated {name}" for name in new_concepts]

        class_to_hyperedge_count = defaultdict(int)
        for edge in hdcm.hyperedges:
            class_to_hyperedge_count[edge.class_name] += 1
        for class_name in classes:
            logger.info(
                f"DCR hyperedges for {class_name}: {class_to_hyperedge_count[class_name]}"
            )
        logger.info(f"DCR hyperedge size: {args.hyperedge_size}")
        logger.info(f"DCR synergy weight: {args.hyperedge_synergy_weight}")
        logger.info(f"DCR hyperedge feature dim: {hdcm_hyperedge_feature_dim}")

        if args.save_hypergraph_structures:
            hypergraph_paths = hdcm.save(
                save_dir,
                split_activations={
                    "train": train_hyperedges,
                    "val": val_hyperedges,
                    "test": test_hyperedges,
                },
            )
            logger.info(f"DCR hypergraph json: {hypergraph_paths['json']}")
            logger.info(f"DCR hypergraph csv: {hypergraph_paths['csv']}")
            logger.info(f"DCR top activated hyperedges csv: {hypergraph_paths['sample_top_csv']}")

    logger.info(f"Feature dim after DCR: {train_c.shape[1]}")
    logger.info(f"Classifier input dim: {train_c.shape[1]}")

    if args.use_dedm:
        dedm_paths = dedm.save_outputs(
            save_dir,
            {
                "train": train_dedm,
                "val": val_dedm,
                "test": test_dedm,
            },
            {
                "train": train_targets,
                "val": val_targets,
                "test": test_targets,
            },
            new_concepts,
            classes,
        )
        logger.info(f"DEDM sample evidence csv: {dedm_paths['sample_csv']}")
        logger.info(f"DEDM class evidence json: {dedm_paths['class_json']}")
        logger.info(f"DEDM class evidence csv: {dedm_paths['class_csv']}")

    # Prepare datasets
    #train_ds = TensorDataset(train_c.cpu(), train_targets)
    #val_ds = TensorDataset(val_c.cpu(), val_targets)
    test_ds = TensorDataset(test_c.cpu(), test_targets)

    train_ds = IndexedTensorDataset(train_c.cpu(), train_targets)
    val_ds = IndexedTensorDataset(val_c.cpu(), val_targets)

    # Train final layer
    linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}
    output_proj = glm_saga(
        linear, 
        DataLoader(train_ds, batch_size=args.saga_batch_size, shuffle=True),
        STEP_SIZE,
        args.n_iters,
        ALPHA,
        epsilon=1, 
        k=1,
        val_loader=DataLoader(val_ds, batch_size=args.saga_batch_size),
        do_zero=False,
        metadata=metadata,
        n_ex=len(train_ds),
        n_classes=len(classes)
    )

    concept_threshold_stats_cpu = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in concept_threshold_stats.items()
    }
    concept_threshold_csv = os.path.join(save_dir, "concept_threshold_stats.csv")
    concept_threshold_json = os.path.join(save_dir, "concept_threshold_stats.json")
    with open(concept_threshold_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_index",
                "concept_name",
                "mu",
                "sigma",
                "tau",
                "valid_count",
            ],
        )
        writer.writeheader()
        mu = concept_threshold_stats_cpu["mu"]
        sigma = concept_threshold_stats_cpu["sigma"]
        tau = concept_threshold_stats_cpu["tau"]
        counts = concept_threshold_stats_cpu["counts"]
        for concept_idx, concept_name in enumerate(new_concepts):
            writer.writerow(
                {
                    "concept_index": concept_idx,
                    "concept_name": concept_name,
                    "mu": float(mu[concept_idx].item()),
                    "sigma": float(sigma[concept_idx].item()),
                    "tau": float(tau[concept_idx].item()),
                    "valid_count": int(counts[concept_idx].item()),
                }
            )
    threshold_records = []
    for concept_idx, concept_name in enumerate(new_concepts):
        threshold_records.append(
            {
                "concept_index": concept_idx,
                "concept_name": concept_name,
                "mu": float(mu[concept_idx].item()),
                "sigma": float(sigma[concept_idx].item()),
                "tau": float(tau[concept_idx].item()),
                "valid_count": int(counts[concept_idx].item()),
            }
        )
    with open(concept_threshold_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "use_soft_concept_labels": args.use_soft_concept_labels,
                "adaptive_threshold": args.adaptive_threshold,
                "lambda_std": args.lambda_std,
                "temperature": args.concept_label_temperature,
                "fixed_threshold": args.Tconf,
                "records": threshold_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Concept threshold stats csv: {concept_threshold_csv}")
    logger.info(f"Concept threshold stats json: {concept_threshold_json}")

    # Save all required components
    save_dict = {
        'cbl_state_dict': best_weights,
        'W_final_layer': output_proj['path'][0]['weight'],
        'b_final_layer': output_proj['path'][0]['bias'],
        'normalization': {
            'train_mean': train_mean.cpu(),
            'train_std': train_std.cpu()
        },
        'concepts': {
            'initial': concepts,
            'filtered': new_concepts,
            'concept_to_classes': concept_to_classes,
            'concept_redirect_map': concept_redirect_map
        },
        'dedm': dedm_state,
        'sdcm': sdcm_state,
        'hdcm': hdcm_state,
        'feature_names': feature_names,
        'args': vars(args),
        'concept_labeling': {
            'use_soft_concept_labels': args.use_soft_concept_labels,
            'adaptive_threshold': args.adaptive_threshold,
            'lambda_std': args.lambda_std,
            'temperature': args.concept_label_temperature,
            'fixed_threshold': args.Tconf,
            'threshold_stats': concept_threshold_stats_cpu,
            'threshold_stats_csv': concept_threshold_csv,
            'threshold_stats_json': concept_threshold_json,
        },
        'concept_labels': {
            'train': concept_labels_train.cpu(),
            'val': concept_labels_val.cpu(),
            'test': concept_labels_test.cpu()
        },
        'concept_binary_labels': {
            'train': concept_binary_labels_train.cpu(),
            'val': concept_binary_labels_val.cpu(),
            'test': concept_binary_labels_test.cpu()
        }
    }
    torch.save(save_dict, os.path.join(save_dir, "full_model.pth"))

    # Save additional human-readable files
    with open(os.path.join(save_dir, "concepts.txt"), 'w') as f:
        f.write("\n".join(new_concepts))
    
    with open(os.path.join(save_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Test evaluation
    W = output_proj['path'][0]['weight'].to(args.device)
    b = output_proj['path'][0]['bias'].to(args.device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in DataLoader(test_ds, batch_size=args.saga_batch_size):
            feats, labels = feats.to(args.device), labels.to(args.device)
            logits = feats @ W.T + b
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    
    test_acc = correct / total
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Save metrics
    metrics = {
        'val_loss': best_val_loss.item(),
        'test_acc': test_acc,
        'sdcm': {
            'enabled': args.use_sdcm,
            'feature_dim_before': len(new_concepts),
            'feature_dim_after': sdcm_feature_dim_after,
            'n_cue_structures': 0 if sdcm_state is None else len(sdcm_state["cue_structures"]),
            'cue_structure_paths': cue_paths,
        },
        'dedm': {
            'enabled': args.use_dedm,
            'num_concept_states': args.num_concept_states,
            'concept_state_names': concept_state_names,
            'uncertainty_weight': args.dedm_uncertainty_weight,
            'kl_weight': args.dedm_kl_weight,
            'uncertainty_summary': dedm_uncertainty_summary,
            'evidence_paths': dedm_paths,
        },
        'hdcm': {
            'enabled': args.use_hdcm,
            'hyperedge_size': args.hyperedge_size,
            'top_hyperedges_per_class': args.top_hyperedges_per_class,
            'hyperedge_activation_type': args.hyperedge_activation_type,
            'hyperedge_binarize_threshold': args.hyperedge_binarize_threshold,
            'hyperedge_synergy_weight': args.hyperedge_synergy_weight,
            'use_hypergraph_message_passing': args.use_hypergraph_message_passing,
            'feature_dim_before': hdcm_feature_dim_before,
            'hyperedge_feature_dim': hdcm_hyperedge_feature_dim,
            'feature_dim_after': train_c.shape[1],
            'n_hyperedges': 0 if hdcm_state is None else len(hdcm_state["hyperedges"]),
            'hypergraph_paths': hypergraph_paths,
        },
        'concept_sparsity': {
            'train': concept_binary_labels_train.float().mean().item(),
            'val': concept_binary_labels_val.float().mean().item(),
            'test': concept_binary_labels_test.float().mean().item()
        },
        'concept_accuracy': concept_accuracy,
        'concept_labeling': {
            'use_soft_concept_labels': args.use_soft_concept_labels,
            'adaptive_threshold': args.adaptive_threshold,
            'lambda_std': args.lambda_std,
            'temperature': args.concept_label_temperature,
            'fixed_threshold': args.Tconf,
            'threshold_stats_csv': concept_threshold_csv,
            'threshold_stats_json': concept_threshold_json,
        },
        'training_metrics': output_proj['path'][0]['metrics']
    }
    with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save concept features and labels
    torch.save({
        'train_features': train_c.cpu(),
        'train_labels': train_targets,
        'val_features': val_c.cpu(),
        'val_labels': val_targets,
        'test_features': test_c.cpu(),
        'test_labels': test_targets,
        'feature_names': feature_names,
        'concept_labels': concept_labels_train.cpu(),
        'concept_binary_labels': concept_binary_labels_train.cpu(),
        'dedm': dedm_state,
        'sdcm': sdcm_state,
        'hdcm': hdcm_state,
    }, os.path.join(save_dir, "concept_data.pt"))

if __name__ == "__main__":
    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("DEBUG")

    parser = argparse.ArgumentParser(description="Train CBM model")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--concept_set", type=str, default="./data/generate_concept/concept/CIFAR10_concepts_gpt-4o_final.json", help="path to concept set name")
    parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

    parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
    parser.add_argument("--cbl_batch_size", type=int, default=512, help="Batch size to use when learning concept bottleneck layer")

    parser.add_argument("--feature_layer", type=str, default='layer4', 
                        help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    parser.add_argument(
        "--use_penultimate", 
        action="store_true",
        default=False,  
        help="Use penultimate layer (default: False)"
    )
    parser.add_argument(
        "--K_indep",
        type=int,
        default=5,
        help="Max number of independent components per class(default: 5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="adam weight_decay"
    )
    parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
    parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
    parser.add_argument("--cbl_steps", type=int, default=20000, help="max steps to train the concept bottleneck layer for")
    parser.add_argument("--cbl_lr", type=float, default=0.001, help="cbl_lr")
    parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
    parser.add_argument("--n_iters", type=int, default=10000, help="How many iterations to run the final layer solver for")
    parser.add_argument("--Tconf", type=float, default=0.20, help="Threshold for filtering and labeling")
    parser.add_argument("--Tmerge", type=float, default=0.9998, help="Threshold for merging concepts")
    parser.add_argument(
        "--use_soft_concept_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use soft concept pseudo-labels for CBL supervision; disable for hard labels."
    )
    parser.add_argument(
        "--adaptive_threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use concept-specific adaptive thresholds; disable to use fixed Tconf."
    )
    parser.add_argument(
        "--lambda_std",
        type=float,
        default=0.5,
        help="Std multiplier for adaptive concept thresholds: tau_j = mu_j + lambda_std * sigma_j."
    )
    parser.add_argument(
        "--concept_label_temperature",
        type=float,
        default=0.1,
        help="Temperature for soft concept labels sigmoid((sim - tau) / temperature)."
    )
    parser.add_argument("--cbl_layer_num", type=int, default=1, help="CBL layer num")
    parser.add_argument("--cbl_bias", action='store_true', help="Use bias in CBL layer")
    parser.add_argument("--seed", type=int, default=42, help="The random seed")
    parser.add_argument("--use_sdcm", action="store_true", help="Enable Structured Diagnostic Cue Modeling")
    parser.add_argument("--top_cue_structures_per_class", type=int, default=5, help="Top diagnostic cue structures to keep per class")
    parser.add_argument("--cue_group_size", type=int, default=2, help="Number of concepts in each cue structure; first version supports 2")
    parser.add_argument("--cue_binarize_threshold", type=float, default=0.5, help="Threshold on sigmoid concept logits for cue mining")
    parser.add_argument("--cue_activation_type", type=str, default="product", choices=["product", "min", "mean"], help="How to compute cue activations from member concepts")
    parser.add_argument("--use_dedm", action="store_true", help="Enable Diagnostic Evidence Distribution Modeling")
    parser.add_argument("--num_concept_states", type=int, default=4, help="Number of diagnostic states for each concept")
    parser.add_argument("--concept_state_names", type=str, default="absent,weak,moderate,strong", help="Comma-separated diagnostic state names")
    parser.add_argument("--dedm_uncertainty_weight", type=float, default=0.01, help="Weight for DEDM uncertainty regularization")
    parser.add_argument("--dedm_kl_weight", type=float, default=0.0, help="Weight for uncertainty-gated KL from concept Dirichlet evidence to a uniform prior")
    parser.add_argument("--use_hdcm", action="store_true", help="Enable Hypergraph Diagnostic Cue Modeling")
    parser.add_argument("--hyperedge_size", type=int, default=3, choices=[2, 3, 4], help="Number of concepts in mined DCR hyperedges")
    parser.add_argument("--top_hyperedges_per_class", type=int, default=5, help="Top DCR hyperedges to keep per class")
    parser.add_argument("--hyperedge_activation_type", type=str, default="product", choices=["product", "min", "mean", "noisy_and"], help="How to compute hyperedge activations")
    parser.add_argument("--hyperedge_binarize_threshold", type=float, default=0.5, help="Threshold for mining active concepts into hyperedges")
    parser.add_argument("--hyperedge_synergy_weight", type=float, default=1.0, help="Weight for synergy-aware DCR scoring: score = discriminativeness + weight * synergy")
    parser.add_argument("--use_hypergraph_message_passing", action="store_true", help="Append one-step hypergraph propagated concept features")
    parser.add_argument("--save_hypergraph_structures", action="store_true", help="Save DCR hypergraph structures and top activated hyperedges")


    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_arg, remaining_args = config_parser.parse_known_args()
    if config_arg.config is not None:
        with open(config_arg.config, "r") as f:
            config_arg = json.load(f)
        parser.set_defaults(**config_arg)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_test_cbm_and_save(args)
