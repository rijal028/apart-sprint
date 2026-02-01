import os
import importlib.util
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# =========================================
# CONFIG
# =========================================
DEFAULT_LAYER_RATIOS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


# =========================================
# HELPERS
# =========================================
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: [1, seq_len, hidden]
    attention_mask: [1, seq_len]
    """
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden_states * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
    return pooled


def load_domain_py_file(filepath):
    """
    File contoh isi:
    DOMAIN_NAME = "biology"
    TEXTS = ["...", "...", ...]
    """
    spec = importlib.util.spec_from_file_location("domain_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    domain_name = getattr(module, "DOMAIN_NAME", None)
    texts = getattr(module, "TEXTS", None)

    if domain_name is None or texts is None:
        raise ValueError(
            f"Invalid domain file: {filepath}\n"
            "Must contain DOMAIN_NAME and TEXTS variables."
        )

    return domain_name, texts


def load_domains_from_folder(domains_dir):
    """
    domains_dir: domains/senior_high/
    expects many .py files each containing DOMAIN_NAME and TEXTS
    """
    domains = {}
    for fname in sorted(os.listdir(domains_dir)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(domains_dir, fname)
        dname, texts = load_domain_py_file(path)
        domains[dname] = texts
    return domains


# =========================================
# ENGINE
# =========================================
class DomainProbeEngine:
    def __init__(self, model_name, local_files_only=True, device=None):
        self.model_name = model_name
        self.local_files_only = local_files_only

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[+] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )

        print(f"[+] Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            local_files_only=local_files_only
        )
        self.model.eval()
        self.model.to(self.device)

        print(f"[+] Model ready on {self.device}")

    def get_embedding(self, text, layer_ratio=1.0, max_length=256):
        """
        layer_ratio=1.0 -> last layer
        layer_ratio=0.4 -> middle-ish layer (buat LLM decoder sering lebih bagus)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        # hidden_states: tuple(len_layers+1), each [1, seq_len, hidden]
        hidden_states = out.hidden_states
        num_layers = len(hidden_states) - 1  # exclude embedding layer

        # choose layer index
        if layer_ratio <= 0:
            layer_ratio = 0.01
        if layer_ratio > 1:
            layer_ratio = 1.0

        layer_index = int(round(num_layers * layer_ratio))
        layer_index = max(1, min(num_layers, layer_index))  # keep valid

        chosen = hidden_states[layer_index]  # [1, seq_len, hidden]
        pooled = mean_pooling(chosen, inputs["attention_mask"])  # [1, hidden]

        # normalize biar cosine stabil
        pooled = torch.nn.functional.normalize(pooled, dim=-1)

        return pooled[0].detach().cpu().numpy()

    def build_prototypes(self, domains, layer_ratio=1.0):
        """
        domains: dict(domain_name -> list_of_texts)
        returns:
          - domain_vectors (prototype)
          - domain_embeddings (per text embedding)
        """
        domain_vectors = {}
        domain_embeddings = {}

        for domain, texts in domains.items():
            embs = [self.get_embedding(t, layer_ratio=layer_ratio) for t in texts]
            domain_embeddings[domain] = embs

            proto = np.mean(embs, axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
            domain_vectors[domain] = proto

        return domain_vectors, domain_embeddings

    def domain_similarity_matrix(self, domain_vectors):
        names = list(domain_vectors.keys())
        matrix = {}

        for d1 in names:
            matrix[d1] = {}
            for d2 in names:
                matrix[d1][d2] = cosine_similarity(domain_vectors[d1], domain_vectors[d2])

        return matrix

    def cohesion_scores(self, domain_vectors, domain_embeddings):
        """
        avg/min/max cosine(text_embedding, centroid)
        """
        cohesion = {}
        for d in domain_vectors.keys():
            centroid = domain_vectors[d]
            sims = [cosine_similarity(e, centroid) for e in domain_embeddings[d]]
            cohesion[d] = {
                "avg": float(np.mean(sims)),
                "min": float(np.min(sims)),
                "max": float(np.max(sims)),
            }
        return cohesion

    def ownership_winrate(self, domain_vectors, domain_embeddings):
        """
        For each text embedding, check nearest centroid.
        """
        names = list(domain_vectors.keys())
        results = {}

        for d in names:
            wins = 0
            total = len(domain_embeddings[d])

            for e in domain_embeddings[d]:
                best_domain = None
                best_score = -1e9

                for cand in names:
                    s = cosine_similarity(e, domain_vectors[cand])
                    if s > best_score:
                        best_score = s
                        best_domain = cand

                if best_domain == d:
                    wins += 1

            results[d] = {
                "wins": wins,
                "total": total,
                "win_rate": float(wins / total) if total > 0 else 0.0
            }

        return results

    def confusion_summary(self, domain_vectors, domain_embeddings, top_k=3):
        """
        Summarize "losses": where each text is misclassified to another domain.
        """
        names = list(domain_vectors.keys())
        summary = {}

        for d in names:
            loss_counts = {}
            total = len(domain_embeddings[d])
            losses = 0

            for e in domain_embeddings[d]:
                # nearest centroid
                best_domain = None
                best_score = -1e9

                for cand in names:
                    s = cosine_similarity(e, domain_vectors[cand])
                    if s > best_score:
                        best_score = s
                        best_domain = cand

                if best_domain != d:
                    losses += 1
                    loss_counts[best_domain] = loss_counts.get(best_domain, 0) + 1

            # top k losses
            top_losses = sorted(loss_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

            summary[d] = {
                "total_losses": losses,
                "total": total,
                "top_losses": top_losses
            }

        return summary

    def global_win_rate(self, ownership_dict):
        """
        Weighted global win-rate
        """
        total_wins = sum(v["wins"] for v in ownership_dict.values())
        total_all = sum(v["total"] for v in ownership_dict.values())
        if total_all == 0:
            return 0.0
        return float(total_wins / total_all)

    def auto_tune_layer(self, domains, candidate_ratios=None):
        """
        Search layer_ratio that maximizes GLOBAL win-rate.
        Useful for decoder LLMs (GPT-Neo, TinyStories, etc.)
        """
        if candidate_ratios is None:
            candidate_ratios = DEFAULT_LAYER_RATIOS

        print("\n=== AUTO-TUNE LAYER SEARCH ===\n")

        best_ratio = None
        best_score = -1

        for r in candidate_ratios:
            domain_vectors, domain_embeddings = self.build_prototypes(domains, layer_ratio=r)
            own = self.ownership_winrate(domain_vectors, domain_embeddings)
            score = self.global_win_rate(own)
            print(f"layer_ratio={r:.2f} -> global_win_rate={score*100:.2f}%")

            if score > best_score:
                best_score = score
                best_ratio = r

        print("\n=== BEST LAYER FOUND ===")
        print(f"best_layer_ratio = {best_ratio:.2f}")
        print(f"best_global_win_rate = {best_score*100:.2f}%")

        return best_ratio, best_score

    def run_full_probe(self, domains, layer_ratio=1.0, do_confusion=True):
        """
        returns dict results
        """
        domain_vectors, domain_embeddings = self.build_prototypes(domains, layer_ratio=layer_ratio)

        sim_matrix = self.domain_similarity_matrix(domain_vectors)
        cohesion = self.cohesion_scores(domain_vectors, domain_embeddings)
        ownership = self.ownership_winrate(domain_vectors, domain_embeddings)

        result = {
            "layer_ratio": layer_ratio,
            "similarity_matrix": sim_matrix,
            "cohesion": cohesion,
            "ownership": ownership,
        }

        if do_confusion:
            result["confusion"] = self.confusion_summary(domain_vectors, domain_embeddings)

        result["global_win_rate"] = self.global_win_rate(ownership)

        return result


# =========================================
# PRINT HELPERS (CLI friendly)
# =========================================
def print_similarity_matrix(sim_matrix):
    print("\n=== DOMAIN SIMILARITY MATRIX ===\n")
    names = list(sim_matrix.keys())

    for d1 in names:
        for d2 in names:
            print(f"{d1:15s} vs {d2:15s}: {sim_matrix[d1][d2]:.4f}")
        print("-" * 45)


def print_cohesion(cohesion):
    print("\n=== DOMAIN COHESION (INTRA-DOMAIN) ===\n")
    for d, stats in cohesion.items():
        print(f"{d:15s} | avg={stats['avg']:.4f} | min={stats['min']:.4f} | max={stats['max']:.4f}")


def print_ownership(ownership):
    print("\n=== DOMAIN OWNERSHIP (WIN-RATE) ===\n")
    for d, stats in ownership.items():
        print(
            f"{d:15s} | wins={stats['wins']:3d}/{stats['total']:3d} | win_rate={stats['win_rate']*100:.2f}%"
        )


def print_confusion(confusion):
    print("\n=== CONFUSION SUMMARY (TOP-3 LOSSES) ===\n")
    for d, info in confusion.items():
        losses = info["total_losses"]
        total = info["total"]

        if losses == 0:
            print(f"{d:15s} | total_losses= 0/{total} | top_losses -> None (perfect ownership)")
            continue

        top_losses_str = ", ".join([f"{x[0]}:{x[1]}" for x in info["top_losses"]])
        print(f"{d:15s} | total_losses={losses:2d}/{total} | top_losses -> {top_losses_str}")