# NoshAI

## Overview

NoshAI is a web-based, backend- and data-driven meal planning system designed to **help people take control of eating and meal preparation regardless of energy level, time, or budget constraints**.

The goal of NoshAI is to empower users to make practical, lower-friction food decisions by turning their existing pantry into a smart assistant that:

* reduces food waste,
* supports balanced nutrition, and
* adapts to real-world constraints like cost, time, and dietary needs.

The system transforms raw pantry inventories and user-defined goals into optimized, explainable meal plans. It prioritizes **interpretability, privacy, and reproducibility**, rather than treating meal planning as a black-box recommendation task.

This repository currently contains the **data pipelines, backend logic, and optimization framework**. Frontend work is under active development and maintained separately.

---

## Problem

Pantry and grocery data is often noisy, inconsistent, and difficult to reason about programmatically (e.g., ingredient variants, missing quantities, unstructured text). Most meal planners either:

* rely on manual user input, or
* treat planning as a pure recommendation problem without transparency.

NoshAI aims to:

* normalize messy pantry data into a structured, machine-readable format
* generate meal plans that are **explainable, constraint-aware, and reproducible**
* balance real-world tradeoffs like cost, time, nutrition, and food waste

---

## Technical Approach

### Data Normalization

* Converts unstructured pantry and ingredient text into standardized representations
* Resolves ingredient variants and synonyms (e.g., naming differences, formats)
* Uses LLM-assisted preprocessing for normalization and summarization, with strict validation to keep downstream logic deterministic

### Optimization Framework

* Models meal planning as a constrained optimization problem rather than a recommendation task
* Balances multiple objectives, including:

  * Cost
  * Cooking time
  * Nutritional quality (FDA macro & micronutrient guidelines)
  * Dietary diversity
  * Food waste reduction
* Optimization logic is fully interpretable and reproducible

### Backend

* Python-based backend services
* Designed for clarity, debuggability, and extensibility
* Separates AI-assisted preprocessing from core optimization logic

---

## Tech Stack

* **Languages:** Python
* **Cloud & Data:** Google Cloud Platform (GCP), BigQuery
* **AI / ML:** Gemini API
* **Frontend (in progress)**: TypeScript, React

---

## Current Status

* Core data pipelines and normalization logic implemented
* Initial optimization framework in place
* Backend APIs under active development
* Frontend integration and end-to-end deployment in progress

---

## Design Principles

* **Clear AI boundaries**: AI is used for data normalization and summarization, while final decisions are made by deterministic logic.
* **Safety by design**: Recommendations stay within user-defined constraints and established nutritional guidelines.
* **Scalable and extensible systems**: Backend and data pipelines are designed to grow with larger datasets and future features.

---

## Next Steps

* Expand ingredient ontology and synonym resolution
* Improve optimization performance and constraint handling
* Integrate frontend and deploy end-to-end system
* Add evaluation metrics for plan quality and constraint satisfaction

---

## Disclaimer

This project is under active development. APIs, data schemas, and optimization logic may evolve as features are added.
