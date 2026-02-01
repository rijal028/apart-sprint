import os
import json

from src.probe_engine import DomainProbeEngine, load_domains_from_folder
from src.visualize_results import export_all_figures


MODEL_NAME = "distilbert-base-uncased"
DOMAINS_DIR = "domains/level 1"
OUTPUT_DIR = "outputs"


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[+] Loading domains from: {DOMAINS_DIR}")
    domains = load_domains_from_folder(DOMAINS_DIR)

    print("[+] Loaded domains:")
    for k in domains.keys():
        print("   -", k)

    engine = DomainProbeEngine(
        model_name=MODEL_NAME,
        local_files_only=True
    )

    best_ratio, best_score = engine.auto_tune_layer(domains)

    result = engine.run_full_probe(domains, layer_ratio=best_ratio, do_confusion=True)

    result_path = os.path.join(OUTPUT_DIR, "results.json")
    save_json(result, result_path)
    print(f"[+] Saved results to: {result_path}")

    export_all_figures(result, out_dir=OUTPUT_DIR)
    print(f"[+] Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

