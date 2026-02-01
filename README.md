# DomainProbe (Apart Sprint Submission)

DomainProbe is a lightweight, offline-capable verification tool that estimates what knowledge domains a language model represents internally  without requiring access to the training dataset.

Instead of evaluating model answers (which can be unstable), DomainProbe extracts internal embeddings from the model and checks whether texts from the same domain form a consistent cluster in representation space.

This project is built for the Apart Research – Technical AI Governance Challenge, focusing on practical capability verification infrastructure.


## WHY THIS MATTERS (TECHNICAL AI GOVERNANCE)


Frontier AI labs may need to demonstrate compliance with safety frameworks and capability thresholds without revealing proprietary training data or model weights.

DomainProbe provides a verification direction:

- Show whether the model internally represents a domain (biology, cybersecurity, etc.)
- Measure how strongly probes belong to their own domain cluster
- Compare encoder vs decoder behavior (important for real-world models)
- Works offline once the model is already downloaded


## CORE IDEA


Probe = training-like forward pass (but not permanent).

During training, text is converted into internal representations (embeddings).
DomainProbe does something similar, but without updating weights:

1) Send domain probe texts into the model (forward pass only)
2) Extract hidden-layer embeddings
3) Pool embeddings into a single vector per text
4) Build domain prototypes (centroids)
5) Compute metrics:
   - Domain Similarity Matrix
   - Domain Cohesion
   - Domain Ownership (Win-rate)

No fine-tuning.
No weight updates.
No dataset exposure.


## OUTPUTS


Running the probe generates:

1) JSON results
- outputs/results.json

Contains:
- similarity matrix
- cohesion metrics
- ownership metrics (win-rate)
- confusion summary (top misclassifications)
- best layer ratio (auto-tuned)

2) Figures (saved automatically)
- outputs/heatmap.png
- outputs/ownership_bar.png
- outputs/cohesion_bar.png


## REPOSITORY STRUCTURE


apart-sprint/

├── run.py

├── src/

│   ├── probe_engine.py

│   └── visualize_results.py

├── domains/

│   └── level_1/ 

│       ├── biology.py 

│       ├── physics.py 

│       ├── chemistry.py

│       ├── mathematics.py

│       ├── computer_science.py

│       ├── cybersecurity.py

│       ├── economics.py

│       ├── medicine.py

│       ├── law.py

│       └── weapons.py

└── outputs/
    ├── results.json
    
    ├── heatmap.png
    
    ├── ownership_bar.png
    
    └── cohesion_bar.png
    
## DOMAIN FILE FORMAT


Each domain file is a simple Python file:

DOMAIN_name = "biology"

TEXTS = [
    "Text 1 ...",
    "Text 2 ...",
    "...",
]


## METRICS


A) Domain Prototypes
Each domain is represented by a prototype vector (centroid):
- embed every probe sentence
- average them into one domain vector

B) Domain Similarity Matrix
Prototype-to-prototype cosine similarity:
- higher = domains are closer internally
- diagonal should be highest (domain vs itself)

C) Domain Cohesion
How compact a domain cluster is:
- avg/min/max cosine similarity between each probe text and its domain prototype

D) Domain Ownership (Win-rate)
For each probe text:
- compare against all prototypes
- a probe wins if its closest prototype is its own domain

Win-rate close to 100% = strong domain separation.


## ENCODER VS DECODER MODELS (IMPORTANT)


This tool supports both types of models:

1) Encoder models (BERT / MPNet)
- typically stable domain separation
- layer choice is less critical

2) Decoder-only LLMs (TinyStories / GPT-Neo)
- final layers often contain more general language mixing
- domain boundaries may blur at later layers

Auto-Tuning Layer Ratio:
DomainProbe automatically searches multiple hidden layers and selects the layer with the best global win-rate.
This makes the method more applicable to decoder LLMs.


## QUICK START


1) Clone repository
git clone https://github.com/rijal028/apart-sprint.git
cd apart-sprint

2) Install dependencies
pip install torch transformers numpy matplotlib

3) Run
python run.py


## CONFIGURATION


Edit run.py:

MODEL_NAME = "distilbert-base-uncased"

DOMAINS_DIR = "domains/level_1"

OUTPUT_DIR = "outputs"

To run fully offline:
- make sure the model is already downloaded locally
- keep local_files_only=True


## LIMITATIONS / NOTES


- This tool does not claim to measure full real-world capability.
- It measures representation alignment between probes and domain clusters.
- Domains may overlap naturally (example: medicine vs biology).
- Results depend on probe text quality and level definitions.

