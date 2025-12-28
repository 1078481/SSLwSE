"""
Enhanced Semi-Supervised Learning (SSL) with Semantic Embeddings
----------------------------------------------------------------------------
Description:
    This script implements "Strategy 3", which integrates medical semantic 
    embeddings (BERT), UMAP dimensionality reduction, and KMeans clustering 
    to enhance the self-training process.

    Process:
    1. Extract features using a local BERT model.
    2. Reduce dimensions via UMAP.
    3. Generate pseudo-labels using KMeans.
    4. Balance pseudo-labels using RandomOverSampler.
    5. Train a SelfTrainingClassifier with a Stacking base model.

Dependencies: pandas, numpy, sklearn, imblearn, transformers, torch, umap-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import warnings

# --- Core Scikit-learn Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.semi_supervised import SelfTrainingClassifier

# --- Imbalanced Learning ---
from imblearn.over_sampling import ADASYN, RandomOverSampler

# --- NLP & Deep Learning ---
from transformers import AutoTokenizer, AutoModel
import torch
import umap  # Ensure 'umap-learn' is installed

# --- Configuration ---
warnings.filterwarnings('ignore')  # Suppress warnings

# Constants for File Paths
DATASET_PATH = '3.xlsx'
SHEET_NAME = 'all-2'
BERT_MODEL_PATH = "./medbert-base-wwm-chinese-local"  # Local path to BERT
BASE_MODEL_TEMPLATE = 'stacking_model_{}.pkl'         # Template for loading base models

# --- 1. Visualization Configuration ---
try:
    config = {
        "font.family": 'serif',
        "font.size": 14,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman', 'DejaVu Serif'],
    }
    plt.rcParams.update(config)
except Exception as e:
    print(f"Warning: Font configuration failed. Using defaults. Error: {e}")


# --- 2. Helper Functions ---

def plot_multiclass_roc(model, X_test, y_test, num_labels, class_names=None, filename="roc.tif"):
    """
    Plots the Multi-class ROC Curve and saves it to a file.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_labels)]
    
    if not hasattr(model, "predict_proba"):
        print(f"Warning: Model {type(model).__name__} lacks 'predict_proba'. Skipping ROC.")
        return

    try:
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        print(f"Error predicting probabilities: {e}")
        return

    y_test_bin = label_binarize(y_test, classes=list(range(num_labels)))
    
    if y_test_bin.shape[1] != y_prob.shape[1]:
        print(f"Error: Label shape {y_test_bin.shape} does not match probability shape {y_prob.shape}.")
        return

    colors = ['red', 'darkorange', 'cornflowerblue']
    linestyles = ["-", "--", ":"]

    plt.figure(figsize=(10, 8))
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        c = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        
        plt.plot(fpr, tpr, color=c, linestyle=ls, linewidth=3,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc:0.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Multi-class ROC Analysis', fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.savefig(filename, dpi=300, format="tif", transparent=True, pad_inches=0.1)
    print(f"ROC figure saved to {filename}")
    plt.close()


# --- 3. Medical Semantic Embedder ---

class MedicalSemanticEmbedder:
    """
    Class to handle text embedding generation using a pre-trained BERT model.
    """
    def __init__(self, model_path):
        print(f"--- Loading BERT model from: {model_path} ---")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' not found.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"--- Model loaded on device: {self.device} ---")

    def clean_text(self, text):
        """Cleans input text by removing special characters."""
        if pd.isna(text):
            return ""
        # Keep Chinese characters, alphanumeric, and standard punctuation
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？,./]', ' ', str(text))
        return text.strip()

    def batch_embed(self, texts, batch_size=32):
        """Generates embeddings for a list of texts in batches."""
        embeddings = []
        cleaned_texts = [self.clean_text(t) for t in texts]
        
        print(f"Embedding {len(cleaned_texts)} texts (Batch Size={batch_size})...")
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the CLS token (first token) as the sentence embedding
                batch_cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_cls_embeddings)
            
            if (i // batch_size) % 10 == 0:
                 print(f"  ... Processed {i + len(batch_texts)} / {len(cleaned_texts)}")

        print("Embedding complete.")
        if not embeddings:
            return np.array([])
        return np.vstack(embeddings)


# --- 4. Main Strategy Execution ---

def run_strategy_3(current_random_state):
    """
    Executes Strategy 3: Enhanced Self-Training.
    """
    # Constants
    RANDOM_STATE = current_random_state 
    TEST_SET_SIZE = 0.2
    N_CLUSTERS = 3
    TEXT_COLUMNS = ['CasInf', 'DocMes', 'PatMes']
    
    # 1. Load Pre-trained Base Model
    print(f"\n--- [Random State: {RANDOM_STATE}] Loading Base Model ---")
    base_model_path = BASE_MODEL_TEMPLATE.format(RANDOM_STATE)
    
    if not os.path.exists(base_model_path):
        print(f"Error: Base model '{base_model_path}' not found. Please ensure stacking models are generated.")
        return None
        
    with open(base_model_path, 'rb') as f:
        base_model = pickle.load(f)

    # 2. Load and Prepare Data
    print(f"--- Loading Data from {DATASET_PATH} ---")
    try:
        data = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Separate Labeled vs Unlabeled
    data_labeled = data[data['PatSat'] != -1].copy()
    data_unlabeled = data[data['PatSat'] == -1].copy()
    
    # Separate Numerical Features vs Text
    X_labeled_num = data_labeled.drop(['PatSat'] + TEXT_COLUMNS, axis=1)
    y_labeled = data_labeled['PatSat']
    X_unlabeled_num = data_unlabeled.drop(['PatSat'] + TEXT_COLUMNS, axis=1)

    # 3. Merge Text Data (for embedding)
    def merge_medical_texts(row):
        parts = []
        # Mapping column names to readable prefixes
        col_map = {'CasInf': 'Case', 'DocMes': 'Doctor', 'PatMes': 'Patient'}
        for col, prefix in col_map.items():
            val = str(row[col]).strip()
            if val not in ['', 'nan', 'None']:
                parts.append(f"{prefix}: {val}")
        return " | ".join(parts) if parts else ""

    unlabeled_texts = data_unlabeled.apply(merge_medical_texts, axis=1).tolist()
    
    num_labels = y_labeled.nunique()
    class_names = ["Neutral", "Positive", "Negative"]

    # 4. Prepare Initial Labeled Data (Train/Val/Test Split)
    # Apply ADASYN to labeled data first
    model_smote = ADASYN(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = model_smote.fit_resample(X_labeled_num, y_labeled)
    
    # Split into Train (Validation for SSL) and Test (Final Evaluation)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train_resampled, y_train_resampled,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_resampled
    )

    # 5. Strategy 3: Generate Pseudo-Labels (BERT + UMAP + KMeans)
    print("\n" + "="*50)
    print("--- Strategy 3: Generating Pseudo-Labels ---")
    print("="*50)

    # A. BERT Embeddings
    embedder = MedicalSemanticEmbedder(BERT_MODEL_PATH) 
    X_unlabeled_emb = embedder.batch_embed(unlabeled_texts)

    # B. UMAP Dimensionality Reduction
    print("Running UMAP (metric='cosine')...")
    reducer = umap.UMAP(
        n_components=20,     # Target dimensions
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',     # Cosine metric for semantic similarity
        random_state=RANDOM_STATE
    )
    X_unlabeled_emb_reduced = reducer.fit_transform(X_unlabeled_emb)
    
    # C. KMeans Clustering
    print("Running KMeans Clustering...")
    k_means = KMeans(
        n_clusters=N_CLUSTERS, 
        init="k-means++", 
        n_init=10, 
        random_state=RANDOM_STATE
    )
    y_pseudo = k_means.fit_predict(X_unlabeled_emb_reduced)
    
    # Check distribution
    unique, counts = np.unique(y_pseudo, return_counts=True)
    print(f"Pseudo-label distribution: {dict(zip(unique, counts))}")

    # D. Balance Pseudo-Labels (Using RandomOverSampler)
    if len(unique) < 2:
        print("Warning: Single cluster detected. Using original unlabeled data without sampling.")
        X_self_augmented = X_unlabeled_num.copy()
    else:
        print("Balancing pseudo-labels using RandomOverSampler...")
        model_ros = RandomOverSampler(random_state=RANDOM_STATE)
        try:
            X_self_augmented, _ = model_ros.fit_resample(X_unlabeled_num, y_pseudo)
            print(f"Data augmented from {len(X_unlabeled_num)} to {len(X_self_augmented)} samples.")
        except ValueError as e:
            print(f"Sampling failed: {e}. Using original data.")
            X_self_augmented = X_unlabeled_num.copy()
    
    # Create labels for self-training (-1 indicates unlabeled)
    y_self_augmented = pd.Series([-1] * len(X_self_augmented), name="PatSat")

    # 6. Combine Data for Self-Training
    # Combine original labeled train set + augmented pseudo-labeled set
    X_ssl_final = pd.concat([X_train_val, X_self_augmented], axis=0, ignore_index=True)
    y_ssl_final = pd.concat([y_train_val, y_self_augmented], axis=0, ignore_index=True)
    model_ada_final = ADASYN(random_state=RANDOM_STATE)
    X_ssl_final, y_ssl_final = model_ada_final.fit_resample(X_ssl_final, y_ssl_final)
    
    # Final Split for the Enhanced Model Training
    X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced = train_test_split(
        X_ssl_final, y_ssl_final,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_ssl_final if len(np.unique(y_ssl_final)) > 1 else None
    )

    # Clean Test Set (Remove any residual -1 labels if they exist)
    combined_test = pd.concat([X_test_enhanced, y_test_enhanced], axis=1)
    combined_test = combined_test[combined_test['PatSat'] != -1]
    X_test_enhanced = combined_test.drop('PatSat', axis=1)
    y_test_enhanced = combined_test['PatSat']

    # 7. Scaling (Using original logic as requested)
    print("Scaling features...")
    scaler_enhanced = StandardScaler()
    X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
    X_test_enhanced_scaled = scaler_enhanced.fit_transform(X_test_enhanced) 

    print(f"Final Strategy 3 Training Set Size: {len(X_train_enhanced_scaled)}")

    # 8. Train Self-Training Classifier
    print("\n--- Training SelfTrainingClassifier (Enhanced) ---")
    ssl_params = {
        'threshold': 0.9,
        'criterion': 'threshold',
        'max_iter': 20,
        'verbose': True
    }
    
    ssl_model_enhanced = SelfTrainingClassifier(base_estimator=base_model, **ssl_params)
    ssl_model_enhanced.fit(X_train_enhanced_scaled, y_train_enhanced)

    # 9. Evaluation
    y_pred_enhanced = ssl_model_enhanced.predict(X_test_enhanced_scaled)
    
    report_enhanced = classification_report(
        y_test_enhanced, y_pred_enhanced, target_names=class_names,
        labels=range(num_labels), digits=4, output_dict=True
    )
    
    run_results = {'SSL_Enhanced': report_enhanced}
    print(f"[SSL_Enhanced] Macro-F1: {report_enhanced['macro avg']['f1-score']:.4f}")

    # 10. Save Artifacts
    print(f"\n--- Saving Models and Plots ---")
    
    # Save Scaler
    with open(f'scaler_enhanced_rs_{RANDOM_STATE}.pkl', 'wb') as f:
        pickle.dump(scaler_enhanced, f)
        
    # Save Model
    with open(f'ssl_enhanced_model_rs_{RANDOM_STATE}.pkl', 'wb') as f:
        pickle.dump(ssl_model_enhanced, f)

    # Save ROC Plot
    roc_fig_multiclass(
        ssl_model_enhanced, 
        X_test_enhanced_scaled, 
        y_test_enhanced, 
        num_labels, 
        class_names, 
        f"roc_ssl_enhanced_rs_{RANDOM_STATE}.tif"
    )

    return run_results


# --- 5. Main Execution Loop ---

if __name__ == "__main__":
    
    random_states_list = [1, 10, 100, 1000, 2025]
    all_run_results = []
    
    print(f"--- [Experiment Start] Running Strategy 3 with {len(random_states_list)} seeds ---")

    for i, rs in enumerate(random_states_list):
        print("\n" + "#"*70)
        print(f"--- Run {i+1}/{len(random_states_list)} (Random State = {rs}) ---")
        print("#"*70)
        
        result = run_strategy_3(current_random_state=rs)
        
        if result is not None:
            all_run_results.append((rs, result))
        else:
            print(f"Skipping RS {rs} due to errors.")
            
    print("\n" + "="*70)
    print("--- [Experiment End] Aggregating Results ---")
    print("="*70)

    # --- Result Aggregation & Reporting ---
    metrics_to_keep = ['precision', 'recall', 'f1-score']
    classes_to_keep = ["Neutral", "Positive", "Negative", "macro avg", "weighted avg"]
    records = []
    
    for rs, run_result in all_run_results:
        for model_name, report in run_result.items():
            if report is None: continue
            
            for class_name in classes_to_keep:
                if class_name in report:
                    record = {
                        'random_state': rs,
                        'model': model_name,
                        'class': class_name
                    }
                    for metric in metrics_to_keep:
                        if metric in report[class_name]:
                            record[metric] = report[class_name][metric]
                    records.append(record)
                    
            if 'accuracy' in report:
                records.append({
                    'random_state': rs,
                    'model': model_name,
                    'class': 'accuracy',
                    'precision': report['accuracy'],
                    'recall': np.nan, 'f1-score': np.nan
                })

    # Save to Excel
    if records:
        results_df = pd.DataFrame(records)
        summary_mean = results_df.groupby(['model', 'class'])[metrics_to_keep].mean()
        summary_std = results_df.groupby(['model', 'class'])[metrics_to_keep].std()
        
        # Flatten 2D data for display
        flat_data = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_mean.values.flat, summary_std.values.flat)]
        reshaped_data = np.array(flat_data).reshape(summary_mean.shape)
        
        summary_combined = pd.DataFrame(
            reshaped_data, index=summary_mean.index, columns=summary_mean.columns
        ).unstack(level='class')
        
        print("--- Final Aggregated Results (Mean ± Std) ---")
        try:
            print(summary_combined.to_markdown())
        except:
            print(summary_combined)

        try:
            output_file = "ssl_strategy3_results.xlsx"
            with pd.ExcelWriter(output_file) as writer:
                results_df.set_index(['random_state', 'model', 'class']).to_excel(writer, sheet_name='Raw_Data')
                summary_mean.to_excel(writer, sheet_name='Mean')
                summary_std.to_excel(writer, sheet_name='Std')
                summary_combined.to_excel(writer, sheet_name='Final_Report')
            print(f"\nSuccess: Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving Excel: {e}")
    else:
        print("No results generated.")