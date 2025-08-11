# -ACM-MM25-EVENTA-TRACK_1-_Rezero-Slavery
# Beyond Vision — Event-Enriched Image Captioning & Retrieval

**Track 1 — EVENTA Grand Challenge (ACM MM 2025)**
**Team:** Re\:zero Slavery — **Top 3 Performer**
Contest: [https://selab.hcmus.edu.vn/events/eventa](https://selab.hcmus.edu.vn/events/eventa)
Paper (PDF): [https://drive.google.com/file/d/1\_-0QwLfkggtv\_6t4AWOEB34\_ba6S5AiL/view](https://drive.google.com/file/d/1_-0QwLfkggtv_6t4AWOEB34_ba6S5AiL/view)&#x20;

---

## Project overview

This repository contains the code, models and utilities used to build a **retrieval-augmented, context-aware image captioning system** for EVENTA Track 1: *Event-Enriched Image Retrieval and Captioning*.
The system fuses visual similarity retrieval, geometric re-ranking, article-context extraction and LLM-based caption refinement to produce long-form, news-style captions that include event background, named entities, outcomes and temporal context — information that cannot be inferred from pixels alone.

Key highlights

* Multi-stage pipeline: semantic image retrieval → visual verification → article chunking → LLM-based caption refinement.
* Ensemble retrieval using BEiT-3 and SigLIP; geometric re-ranking via ORB+SIFT.
* Base captions from InstructBLIP (Vicuna-7B); context-aware refinement via QLoRA-finetuned DeepSeek-R1 (Qwen3-8B).
* Achieved **Top 3** on EVENTA Track 1 (Overall scores reported in paper). See the paper for full metrics and analysis.&#x20;

---

## Repository structure (high-level)

```
.
├── ImageRetrieval/
│   ├── retrieval_system.py        # main retrieval pipeline (BEiT-3 + SigLIP + ensemble)
│   ├── rerank_code.py             # geometric re-ranking using ORB/SIFT + scoring
│   ├── docker-compose.yml         # docker config (optional)
│   ├── database_images_to_article_*.json
│   ├── database_article_to_url.json
│   └── README.md (local)
├── VideoCrawler/                  # ancillary scripts used during dataset preparation / caption generation
│   ├── create-base-captions.ipynb
│   ├── run-inference-private.ipynb
│   ├── post_process_optimize_cider.py
│   └── other utilities / logs
├── retrieval_results/             # example retrieval outputs and cached features
├── top10_articles.csv             # example retrieved articles per query (top-10)
├── post_process_fix_error_captions.py
├── README.md                      # THIS FILE
└── Track1-Top3.pdf                # paper describing method & results (submitted)
```

---

## System architecture (conceptual)

1. **Visual embedding & retrieval**

   * Generate image embeddings with BEiT-3 and SigLIP2.
   * Fuse/ensemble embeddings and retrieve top candidates (top-2 from each model, weighted ensemble).
2. **Geometric re-ranking**

   * Use ORB + SIFT keypoint matching, Lowe’s ratio test and RANSAC to compute inlier ratios and spatial/scale consistency.
   * Compute final confidence score and re-rank candidates.
3. **Article chunking & semantic selection**

   * Split candidate articles into sentence chunks (sliding window) and embed chunks with `all-MiniLM-L12-v2`.
   * Use cosine similarity between base caption and chunks to select the most relevant text segments.
4. **Caption generation & refinement**

   * Generate a detailed **base caption** with InstructBLIP (Vicuna-7B) using a journalism-focused prompt.
   * Concatenate selected article chunks + base caption into a task prompt.
   * Fine-tune a DeepSeek→Qwen3-8B model with QLoRA for context-enriched caption generation (prompting rules emphasize WHO/WHAT/WHY/WHEN/WHERE).
5. **Post-processing**

   * Re-ranking candidate captions, fluency polishing, and format/CSV submission generation.

For full technical details and ablation results, see our paper.&#x20;

---

## Requirements & recommended environment

Minimum (example)

* Python 3.9+
* PyTorch (compatible CUDA)
* transformers, accelerate
* sentence-transformers
* faiss-cpu / faiss-gpu
* OpenCV (`opencv-python`)
* scikit-learn, numpy, pandas
* tokenizers, datasets (optional)
* Docker & docker-compose (optional — a `docker-compose.yml` exists)

Example installation (conda):

```bash
conda create -n eventa python=3.9 -y
conda activate eventa
pip install -r requirements.txt
# If no requirements file, install these:
pip install torch torchvision torchaudio
pip install transformers accelerate sentence-transformers faiss-cpu opencv-python scikit-learn pandas numpy
```

---

## Quick start — reproducing core pipeline (simplified)

> **Important:** This repo expects precomputed visual/article indices and (optionally) model files. Full reproduction requires access to OpenEvents v1 or the EVENTA dataset and sufficiently large compute (GPU) for model inference / fine-tuning.

1. **Prepare dataset & indices**

   * Place images and article DB JSON files under `ImageRetrieval/`.
   * Build or load precomputed image embeddings for BEiT-3 and SigLIP2. (Scripts to compute embeddings are located in `ImageRetrieval/` — adapt model names & batch sizes.)

2. **Run retrieval** (semantic + ensemble)

```bash
python ImageRetrieval/retrieval_system.py \
  --images_dir /path/to/images \
  --beitz_model beit3-model-name \
  --siglip_model siglip-model-name \
  --index_path /path/to/faiss.index \
  --topk 10 \
  --out_dir retrieval_results/
```

3. **Geometric reranking**

```bash
python ImageRetrieval/rerank_code.py \
  --retrieval_dir retrieval_results/ \
  --query_dir /path/to/query_images \
  --out reranked_results.json
```

4. **Generate base captions (InstructBLIP)**

* Use notebook `VideoCrawler/create-base-captions.ipynb` or call script to generate base captions per image. Save output mapping `query_id -> base_caption`.

5. **Semantic chunk selection & QLoRA inference**

* Use chunking + `all-MiniLM-L12-v2` to pick top article chunks.
* Provide concatenated prompt to the QLoRA-finetuned model and run inference to generate enriched captions.

6. **Post-process & create submission**

```bash
python post_process_fix_error_captions.py --captions_file enriched_captions.json --out_csv submission.csv
# Zip the submission.csv as submission.zip and upload to CodaBench per EVENTA rules.
```

---

## Submission format (EVENTA Track 1)

* CSV named `submission.csv` (no blank lines), zipped to `submission.zip`.
* 12 columns: `query_id, article_id_1, ..., article_id_10, generated_caption`
* If an article cannot be retrieved, use `#` as placeholder.
* Example line:

```
12312,56712,56723,56734,56745,56756,56767,56778,56789,56790,56701,"A group of children playing soccer on a sunny afternoon."
```

Refer to EVENTA submission instructions for precise formatting and limits.

---

## Evaluation metrics (official)

The EVENTA Track 1 leaderboard uses a combined **Overall Score** computed as a weighted harmonic mean of retrieval and captioning metrics:

* Retrieval: AP (Average Precision), Recall\@1, Recall\@10
* Captioning: CLIPScore, CIDEr
  Default weights (used by EVENTA): AP=1, Recall\@1=2, Recall\@10=2, CLIPScore=3, CIDEr=2.
  For full formula and evaluation details, see the EVENTA website and the challenge description.

---

## Results (selected)

Reported private test metrics (hidden test) from challenge:

* **Overall score:** 0.45148
* **AP:** 0.955 | **Recall\@1:** 0.945 | **Recall\@10:** 0.973
* **CLIPScore:** 0.732 | **CIDEr:** 0.156

Public test metrics (for comparability):

* **Overall:** 0.51815 | **AP:** 0.994 | **Recall\@1:** 0.990 | **Recall\@10:** 1.000 | **CLIPScore:** 0.748 | **CIDEr:** 0.195

(See the paper for full tables, comparisons with baselines and ablations.)&#x20;

---

## Reproducibility notes & tips

* **Data access:** EVENTA/OpenEvents v1 is large and may require registration. Use the official challenge data for exact replication.
* **Pretrained models:** BEiT-3, SigLIP2, InstructBLIP, Qwen3 / DeepSeek-R1 weights are referenced in the paper. Use the same model checkpoints where possible or compatible open alternatives.
* **Indexing & FAISS:** Use consistent embedding normalization and index parameters to reproduce retrieval performance. Save both index & ID mappings.
* **QLoRA fine-tuning:** Fine-tuning settings (rank, epochs, quantization, prompt engineering) materially affect caption quality. See the paper for hyper-parameters (QLoRA rank=256, 8-bit, 2 epochs in our runs).&#x20;
* **Hardware:** Reranking + LLM inference benefits from GPU(s); QLoRA fine-tuning can be done on a single high-memory GPU using 8-bit adapters, but time and memory requirements vary.

---

## Ablations & design choices (short summary)

* **Ensembling BEiT-3 + SigLIP2** improved semantic recall vs single model retrieval.
* **Geometric re-ranking (ORB + SIFT)** reduces false positives from visually similar but contextually different images.
* **Semantic chunk selection** (all-MiniLM-L12-v2) helps LLM focus on the most informative article segments, improving CIDEr and CLIPScore.
* **QLoRA fine-tuning** of DeepSeek-R1 enables stronger reasoning and integration of article context while remaining efficient.

For full experimental tables and analysis, consult the paper.&#x20;

---

## How to cite this work

If you use or build upon this repository, please cite our paper:

> *Beyond Vision: Contextually Enriched Image Captioning with Multi-Modal Retrieval.* Re\:zero Slavery, EVENTA (ACM Multimedia 2025). PDF: [https://drive.google.com/file/d/1\_-0QwLfkggtv\_6t4AWOEB34\_ba6S5AiL/view](https://drive.google.com/file/d/1_-0QwLfkggtv_6t4AWOEB34_ba6S5AiL/view).&#x20;

(Include the full citation record in publications that report or compare with our results.)

---

## License

This repository is released under the **MIT License** — see `LICENSE` (create one if not present).

> Note: third-party pretrained model weights and datasets may have their own licenses and usage restrictions. Be sure to comply with each model / dataset license.

---

## Acknowledgements & references

* EVENTA Grand Challenge organizers and dataset providers. [https://selab.hcmus.edu.vn/events/eventa](https://selab.hcmus.edu.vn/events/eventa)
* Pretrained models and libraries: BEiT-3, SigLIP2, InstructBLIP, Qwen3 / DeepSeek, Sentence-Transformers, FAISS, Hugging Face Transformers.
* See our paper for a full list of references and a complete methodological discussion.&#x20;

---


### Need help reproducing a specific experiment?

Tell us:

1. Which step you need (retrieval / rerank / chunking / fine-tuning / inference).
2. Your hardware (GPU model and memory).
3. Whether you have the EVENTA/OpenEvents v1 dataset or need assistance with dataset subset / sample generation.

We can provide focused scripts and parameter settings to help reproduce the results in the paper.
