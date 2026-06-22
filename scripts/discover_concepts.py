import argparse
import base64
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.exemplar_selection import select_exemplars


def clean_readable_name(class_name: str) -> str:
    name = re.sub(r"^\d+\.", "", class_name)
    return name.replace("_", " ").strip()


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_prompt(readable_name: str, dataset_name: str, sample_strategy: str) -> List[Dict]:
    return [
        {
            "type": "text",
            "text": f"""
You are shown sample medical images from the class "{readable_name}" in the dataset "{dataset_name}".

These images were automatically selected from this class's training distribution using the "{sample_strategy}" strategy.
They include distribution-aware representative samples and, when available, boundary samples that are visually close to other classes.

Your task is to summarize stable medical visual cues for this class in this current dataset.
Do not list generic medical textbook knowledge unless it is visible in these images.
Focus on concrete, recurring image-level evidence: color, pigment pattern, border shape, texture, local structure, lesion geometry, and visible artifacts.

Each concept must be short, concrete, visually checkable, and concise (<=35 characters).
Output a plain list only, without numbering or full sentences.

Output format example:
-irregular border
-dark central region
-mottled pigmentation
-rough scaly surface
-visible vascular pattern

Now generate the feature list:
""",
        }
    ]


def chunk_images(image_files: List[Path], chunks: int = 2) -> List[List[Path]]:
    if not image_files:
        return []
    chunks = min(chunks, len(image_files))
    return [image_files[i::chunks] for i in range(chunks)]


def clean_concepts(raw_output: str, max_len: int) -> List[str]:
    concepts = []
    for line in raw_output.strip().splitlines():
        concept = line.strip(" -\t")
        concept = re.sub(r"^\d+[\).\s-]*", "", concept).strip()
        if concept and len(concept) <= max_len:
            concepts.append(concept)
    return concepts


def generate_concepts(
    dataset_name: str,
    selected_root: str,
    concept_output_path: Path,
    gpt_model: str,
    sample_strategy: str,
    max_images: int,
    max_concept_len: int,
    api_key: str,
    base_url: str,
) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for concept generation. Install it with `pip install openai`.") from exc

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    selected_root_path = Path(selected_root) / dataset_name
    class_names = [d.name for d in selected_root_path.iterdir() if d.is_dir()]
    all_concepts = {}

    for class_name in sorted(class_names):
        image_dir = selected_root_path / class_name
        image_files = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )[:max_images]
        if not image_files:
            continue

        readable_name = clean_readable_name(class_name)
        all_responses = []
        for split_idx, split_files in enumerate(chunk_images(image_files, chunks=2), start=1):
            message_content = build_prompt(readable_name, dataset_name, sample_strategy)
            for image_path in split_files:
                base64_img = encode_image(image_path)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    }
                )

            try:
                completion = client.chat.completions.create(
                    model=gpt_model,
                    messages=[{"role": "user", "content": message_content}],
                )
                raw_output = completion.choices[0].message.content
                all_responses.extend(clean_concepts(raw_output, max_concept_len))
            except Exception as exc:
                print(f"[concept] failed on {class_name} split {split_idx}: {exc}")

        unique_concepts = sorted(set(all_responses))
        all_concepts[readable_name] = unique_concepts
        print(f"[concept] {readable_name}: {unique_concepts}")

    concept_output_path.parent.mkdir(parents=True, exist_ok=True)
    with concept_output_path.open("w", encoding="utf-8") as f:
        json.dump(all_concepts, f, indent=2, ensure_ascii=False)
    print(f"[concept] init path: {concept_output_path}")


def reorder_by_class_list(concept_output_path: Path, ordered_output_path: Path, class_list_path: str) -> None:
    with open(class_list_path, "r", encoding="utf-8") as f:
        ordered_classes = [line.strip().replace("_", " ") for line in f if line.strip()]
    with concept_output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ordered_data = OrderedDict()
    for cls in ordered_classes:
        if cls in data:
            ordered_data[cls] = data[cls]

    with ordered_output_path.open("w", encoding="utf-8") as f:
        json.dump(ordered_data, f, indent=2, ensure_ascii=False)
    print(f"[concept] ordered path: {ordered_output_path}")


def convert_to_key_value_lists(input_path: Path) -> Dict[str, List[Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    value_list = []
    key_list = []
    for i, (_, values) in enumerate(data.items()):
        for val in values:
            value_list.append(val)
            key_list.append(i)
    return {"concepts": value_list, "concepts_to_class": key_list}


def save_key_value_format(ordered_output_path: Path, final_output_path: Path) -> None:
    output = convert_to_key_value_lists(ordered_output_path)
    with final_output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[concept] final path: {final_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribution-aware concept discovery for PS-CBM.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--sample_strategy",
        choices=[
            "random",
            "diverse",
            "representative",
            "representative_only",
            "boundary",
            "boundary_only",
            "hybrid",
            "representative_boundary",
            "representative+boundary",
        ],
        default="random",
    )
    parser.add_argument("--num_exemplars", type=int, default=8)
    parser.add_argument("--clip_name", type=str, default="biomedclip")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected_root", type=str, default="data/selected_image")
    parser.add_argument("--embedding_cache_dir", type=str, default="data/embedding_cache")
    parser.add_argument("--activation_dir", type=str, default="", help="Optional existing PS-CBM activation cache to reuse.")
    parser.add_argument("--concept_output_dir", type=str, default="data/generate_concept/concept")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--max_concept_len", type=int, default=35)
    parser.add_argument("--skip_gpt", action="store_true", help="Only select exemplars and skip VLM concept generation.")
    args = parser.parse_args()

    metadata = select_exemplars(
        dataset_name=args.dataset,
        split=args.split,
        clip_name=args.clip_name,
        sample_strategy=args.sample_strategy,
        num_exemplars=args.num_exemplars,
        output_root=args.selected_root,
        cache_dir=args.embedding_cache_dir,
        activation_dir=args.activation_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    print(f"[concept] sample strategy: {args.sample_strategy}")
    print(f"[concept] embedding cache path: {metadata['embedding_cache']}")

    suffix = f"{args.gpt_model}_{args.sample_strategy}"
    concept_output_path = Path(args.concept_output_dir) / f"{args.dataset}_concepts_{suffix}_init.json"
    ordered_output_path = Path(args.concept_output_dir) / f"{args.dataset}_concepts_{suffix}_ordered.json"
    final_output_path = Path(args.concept_output_dir) / f"{args.dataset}_concepts_{suffix}_final.json"
    class_list_path = f"data/classes_name/{args.dataset}_classes.txt"

    if args.skip_gpt:
        print("[concept] skip_gpt is set; no concept JSON was generated.")
        print(f"[concept] intended final path: {final_output_path}")
        return
    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api_key.")

    generate_concepts(
        dataset_name=args.dataset,
        selected_root=args.selected_root,
        concept_output_path=concept_output_path,
        gpt_model=args.gpt_model,
        sample_strategy=args.sample_strategy,
        max_images=args.max_images,
        max_concept_len=args.max_concept_len,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    reorder_by_class_list(concept_output_path, ordered_output_path, class_list_path)
    save_key_value_format(ordered_output_path, final_output_path)


if __name__ == "__main__":
    main()
