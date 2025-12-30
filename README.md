# A Theory-driven Machine Learning Design Approach for Predicting Patient Satisfaction with Online Consultation Services

## Overview

The project implements a comparative study of machine learning strategies for medical text prediction, specifically focusing on patient satisfaction (`PatSat`). It compares:

1. **Baseline Supervised Models**: Stacking Ensemble (trained on labeled data) (Strategy 1) and some widely used machine learning models.
2. **Standard Self-Training**: Traditional semi-supervised learning (Strategy 2).
3. **Proposed Method (Ours)**: Enhanced Self-Training utilizing BERT embeddings, UMAP dimensionality reduction, and KMeans clustering (Strategy 3).

## Directory Structure

Please ensure your project directory is organized as follows before running the scripts:

```text
Project_Root/
│
├── data/
│   ├── 1.xlsx                        # Dataset for Baselines & Standard Self-Training
│   └── 3.xlsx                        # Dataset for Proposed Method (Ours)
│
├── models/                           # Directory to store trained models (.pkl)
│   └── (Generated during runtime)
│
├── medbert-base-wwm-chinese-local/   # [Required] Local BERT Model Directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
│
├── scripts/
│   ├── baselines&stacking.py        # Code for baselines & Strategy 1
│   ├── self_stacking.py        # Code for Strategy 2
│   └── ours.py      # Code for Strategy 3 (Ours)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

```

---

## 1. Environment Setup

The code requires **Python 3.8+**. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt

```

**Note on `umap-learn`:** The code uses UMAP for dimensionality reduction. Ensure it is installed via `pip install umap-learn` (not `umap`).

---

## 2. Data Preparation

This framework relies on two specific Excel files. Please place them in the project root or update the `DATASET_PATH` variable in the scripts accordingly.

### **File 1: `1.xlsx**`

* **Usage:** Used for training **Baseline Models (Stacking)** and **Standard Self-Training (Strategy 2)**.
* **Format:** Excel file containing columns:
* Numerical Features (various)
* `PatSat` (Target variable: 0, 1, 2 for labeled; -1 for unlabeled)



### **File 2: `3.xlsx**`

* **Usage:** Used exclusively for **The Proposed Method (Strategy 3 / Ours)**.
* **Format:** Must strictly follow the same column structure as `1.xlsx`.
* `CasInf`, `DocMes`, `PatMes` (Text columns)
* `PatSat` (Target variable: 0, 1, 2 for labeled; -1 for unlabeled)


---

## 3. Model Preparation (Crucial)

The proposed method utilizes a pre-trained Chinese BERT model (`medbert-base-wwm-chinese`). **You must download this model locally before running Strategy 3.**

### How to Download

1. Visit the Hugging Face model page: [trueto/medbert-base-wwm-chinese](https://huggingface.co/trueto/medbert-base-wwm-chinese) (or any other compatible medical BERT).
2. Download the following files:
* `config.json`
* `pytorch_model.bin`
* `tokenizer.json` (or `tokenizer_config.json`)
* `vocab.txt`


3. Place these files inside a folder named `medbert-base-wwm-chinese-local` in the project root.

*Alternatively, run this Python snippet to download it automatically:*

```python
from transformers import AutoTokenizer, AutoModel
save_directory = "./medbert-base-wwm-chinese-local"
model_name = "trueto/medbert-base-wwm-chinese" # Example model name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print(f"Model saved to {save_directory}")

```

---

## 4. Usage Instructions

### Step 1: Train Base Stacking Model & Standard Self-Training

This script trains the initial Stacking Classifier and performs the standard Self-Training comparison using `1.xlsx`.

```bash
python scripts/baselines&stacking.py

```

```bash
python scripts/self_stacking.py

```

* **Input:** `1.xlsx`
* **Output:** `stacking_model_SEED.pkl/self_stacking_model_SEED.pkl`

### Step 2: Run Proposed Method (Ours)

This script executes the enhanced semi-supervised pipeline (BERT + UMAP + KMeans) using `3.xlsx`.

**Important:** Ensure the variable `DATASET_PATH` in this script is set to `'3.xlsx'`.

```bash
python scripts/ours.py

```

* **Input:** `3.xlsx` (and pre-trained `stacking_model_SEED.pkl` from Step 1)
* **Output:** * Excel Report: `ssl_strategy3_results.xlsx`
* ROC Curves: `roc_ssl_enhanced_SEED.tif`
* Saved Models: `ssl_enhanced_model_SEED.pkl`



---

## 5. Outputs

After execution, the following results are generated:

1. **Excel Reports:** Comprehensive tables containing Precision, Recall, and F1-score (Mean ± Std) across 5 random seeds.
2. **ROC Curves:** High-resolution `.tif` images (300 DPI) suitable for publication.
3. **Model Files:** Serialized `.pkl` files for reproducibility.


---

## 6. Using the Pre-trained Model for Reasoning
We provide a production-ready checkpoint so that users can directly use the model for reasoning and verify the effectiveness of the aircraft logic. This allows you to skip the training phase and strictly evaluate the performance of Strategy 3.

### 1. Preparation

Ensure you have the following files in your project directory:

* **Model File:** `ssl_enhanced_model_rs_2025.pkl` (The pre-trained model weights.).
* **Scaler File:** `scaler_enhanced_rs_2025.pkl` (Required to normalize new data exactly as done during training.).
* **Inference Dataset:** `3.xlsx` (The data file for inference.).

### 2. Run Inference
We have prepared a script to load the model and process the aircraft data. Create a file named ours_inference.py and execute the following command:

```bash
python ours_inference.py

```

### Note

* This script assumes the `ssl_enhanced_model_rs_2025.pkl` was saved using the `pickle` library as shown in our training scripts.
* **No BERT Required:** Since the final classifier operates on numerical features (the semantic embeddings were only used to generate pseudo-labels during the training phase), you do not need to download the BERT model to run this specific testing script.


## Contact

For any questions regarding the code or dataset, please contact the author.

