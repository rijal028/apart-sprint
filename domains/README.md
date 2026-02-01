# Domains Folder

This folder contains the **domain probe datasets** used by DomainProbe.

Each domain dataset is a lightweight list of factual probe texts written in plain language.
The goal is to test whether a model internally represents each domain as a separable cluster in embedding space.


## STRUCTURE


domains/
 
└── level_1/

    ├── biology.py
    
    ├── physics.py
    
    ├── chemistry.py
    
    ├── mathematics.py
    
    ├── computer_science.py
    
    ├── cybersecurity.py
    
    ├── economics.py
    
    ├── medicine.py
    
    ├── law.py
    
    └── weapons.py


## LEVELS


Levels represent increasing difficulty / specialization of probe texts.

Example plan:
- level_1 = foundational knowledge (school-level core concepts)
- level_2 = undergraduate / professional-level fundamentals
- level_3 = advanced / research-level probes

(Only level_1 is included by default in this repo version.)


## DOMAIN FILE FORMAT


Each domain file is a Python module containing:

1) DOMAIN_name (string)
2) TEXTS (list of strings)

Example:

DOMAIN_name = "biology"

TEXTS = [
    "DNA replication produces two identical DNA molecules before cell division.",
    "Transcription converts DNA into messenger RNA.",
    "..."
]


## ADDING NEW DOMAINS


To add a new domain:
1) Create a new file inside the selected level folder (example: domains/level_1/)
2) Follow the same format: DOMAIN_name + TEXTS
3) Run the probe using run.py


## NOTES


- Probe texts should be factual and domain-specific.
- Some overlap between domains is expected (example: medicine and biology).
- Higher-quality probes produce clearer domain ownership separation.
