# Semantic Trust Boundary Violation (STBV)
## A Multi-Layer Trust Architecture for Secure V2X Communication

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub Actions](https://img.shields.io/github/actions/workflow/status/Mukilskanda/semantic-trust-boundary-violation/ci.yml?branch=main)

---

# Overview

This repository implements a complete Secure V2X Trust Stack designed to detect and mitigate **Semantic Trust Boundary Violations (STBV)** in Cooperative Intelligent Transportation Systems (C-ITS).

Unlike existing architectures that validate cryptographic authenticity but implicitly trust inter-layer outputs, this work introduces a semantic trust layer capable of identifying prompt-injection style semantic attacks and malicious cooperative messages.

The implemented pipeline combines:

- Public Key Infrastructure (PKI)
- Secure Cryptographic Validation (B1)
- Misbehavior Detection (MBD)
- Explainability Layer (B2)
- Cooperative Perception (CP)
- Semantic Trust Analysis (B3)
- Trust Decision Engine
- Dempster–Shafer Adapter Layer

forming a complete end-to-end secure V2X trust architecture.

---

# Final Architecture

```
            V2X Messages
                  │
                  ▼
        ┌────────────────────┐
        │ Public Key         │
        │ Infrastructure     │
        │ (PKI)              │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ B1                 │
        │ Secure Cryptographic
        │ Validation         │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ MBD                │
        │ Misbehavior        │
        │ Detection          │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ B2                 │
        │ Explainability     │
        │ Layer              │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ CP                 │
        │ Cooperative        │
        │ Perception         │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ B3                 │
        │ Semantic Trust     │
        │ Gate               │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Trust Decision     │
        │ Engine             │
        └────────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Adapters           │
        │ Logging/API/DS     │
        └────────────────────┘
                  │
                  ▼
          Final Trust Decision
```

---

# Repository Structure

```
b1_scsv/               Secure Cryptographic Validation

pki/                   Public Key Infrastructure

mbd/                   Misbehavior Detection

b2_explain/            Explainability Layer

cp/                    Cooperative Perception

b3/                    Semantic Trust Layer

trust_engine/          Trust Fusion Engine

adapters/              Logging/API/Dempster–Shafer adapters

pipeline/              Complete execution pipeline

tests/                 Unit and validation tests

test_messages/         Standard message fixtures

scenarios/             Attack scenarios

validation/            Validation utilities
```

---

# Features

✔ PKI validation

✔ Certificate verification

✔ Signature verification

✔ Replay detection

✔ Timestamp validation

✔ Misbehavior detection

✔ Explainability generation

✔ Cooperative perception

✔ Semantic attack detection using DeBERTa

✔ Trust fusion

✔ Dempster–Shafer adapter

✔ Multiple attack simulations

✔ GitHub CI

---

# Installation

Clone the repository

```bash
git clone https://github.com/Mukilskanda/semantic-trust-boundary-violation.git

cd semantic-trust-boundary-violation
```

Create virtual environment

Linux / WSL

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Single message

```bash
python3 manual_pipeline_test.py \
--pipeline test_messages/benign/normal_car.json \
--verbose
```

Benign regression

```bash
python3 manual_pipeline_test.py \
--pipeline test_messages/benign
```

Invalid message regression

```bash
python3 manual_pipeline_test.py \
--pipeline test_messages/b1_fail
```

Urban context regression

```bash
python3 manual_pipeline_test.py \
--pipeline test_messages/context/urban
```

---

# Running Attack Scenarios

Sybil

```bash
python3 manual_pipeline_test.py \
--pipeline scenarios/sybil
```

Replay

```bash
python3 manual_pipeline_test.py \
--pipeline scenarios/replay
```

Collusion

```bash
python3 manual_pipeline_test.py \
--pipeline scenarios/collusion
```

Fabrication

```bash
python3 manual_pipeline_test.py \
--pipeline scenarios/fabrication
```

Mixed

```bash
python3 manual_pipeline_test.py \
--pipeline scenarios/mixed
```

---

# B3 Semantic Model Verification

Verify that the DeBERTa semantic model loads correctly.

```bash
python3 tests/verify_b3_model.py
```

Expected output

```
Available : True

Label : BENIGN

Confidence : 0.99+
```

---

# Unit Tests

Dependency graph

```bash
python3 tests/verify_dependency_graph.py
```

Trust Engine

```bash
python3 tests/test_b2_trust_engine.py
```

Adapters

```bash
python3 tests/test_adapters.py
```

---

# GitHub Actions

Every push automatically runs

- Dependency graph verification
- Trust Engine tests
- Adapter tests
- Regression suites
- Attack scenario validation
- Semantic model verification

---

# Research Contributions

This work proposes a Semantic Trust Boundary Violation (STBV) framework capable of identifying attacks that bypass traditional cryptographic validation while manipulating downstream semantic reasoning.

Major contributions include:

- Multi-layer trust architecture
- Explainable trust reasoning
- Semantic trust validation
- Trust fusion engine
- Dempster–Shafer evidence adaptation
- End-to-end V2X security pipeline

---

# Citation

If this work contributes to your research, please cite:

```
@misc{stbv2026,
  title={Semantic Trust Boundary Violation Detection in Secure V2X Communication},
  author={Mukil Skanda},
  year={2026}
}
```

---

# License

MIT License

---

# Author

Mukil Skanda

PES University

Bengaluru, India

---

# Acknowledgements

This project builds upon concepts from:

- ETSI ITS standards
- IEEE 1609 WAVE
- Cooperative Intelligent Transportation Systems (C-ITS)
- Transformer-based Semantic Analysis
- Dempster–Shafer Evidence Theory