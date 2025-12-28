"""
Semi-Supervised Learning Comparative Analysis
-------------------------------------------------------------------------
Description:
    This script evaluates and compares two strategies for classification 
    in a semi-supervised setting:
    1. Baseline: A pre-trained Stacking Ensemble model (Supervised).
    2. Self-Training: A semi-supervised SelfTrainingClassifier using the 
       Stacking model as the base estimator.

    The experiment is repeated over multiple random seeds to ensure 
    statistical robustness.

Strategies:
    - Strategy 1 (Baseline): Direct evaluation of the pre-trained model.
    - Strategy 2 (Self-Training): Standard self-training on labeled + unlabeled data.

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

# --- Core Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, roc_curve, auc, f1_score

# --- Imbalanced Learning ---
from imblearn.over_sampling import ADASYN

# --- Models ---
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import SelfTrainingClassifier
import xgboost as xgb

# --- Configuration ---
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# File Paths (Update these as necessary)
DATASET_PATH = '1.xlsx'
SHEET_NAME = 'all-2'
OUTPUT_EXCEL = 'ssl_comparison_report.xlsx'

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
    print(f"Warning: Error setting font configurations. Using default fonts. Error: {e}")


# --- 2. Helper Functions ---

def plot_multiclass_roc(model, X_test, y_test, num_labels, class_names=None, filename="roc.tif"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for multi-class classification.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_labels)]
    
    # Check if model supports probability prediction
    if not hasattr(model, "predict_proba"):
        print(f"Warning: Model {type(model).__name__} does not support 'predict_proba'. Skipping ROC.")
        return

    try:
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        print(f"Error predicting probabilities: {e}")
        return

    y_test_bin = label_binarize(y_test, classes=list(range(num_labels)))
    
    if y_test_bin.shape[1] != y_prob.shape[1]:
        print(f"Error: Mismatch in class count. Labels: {y_test_bin.shape[1]}, Probs: {y_prob.shape[1]}")
        return

    colors = ['red', 'darkorange', 'cornflowerblue']
    linestyles = ["-", "--", ":"]

    plt.figure(figsize=(10, 8))
    
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Handle color cycling if classes > 3
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


# --- 3. Main Experiment Function ---

def run_ssl_comparison(current_random_state):
    """
    Executes the comparative experiment for a specific random seed.
    Compares Strategy 1 (Baseline) vs Strategy 2 (Self-Training).
    """
    
    # Constants
    RANDOM_STATE = current_random_state 
    TEST_SET_SIZE = 0.2
    
    print(f"\n--- [Random State: {RANDOM_STATE}] Loading Baseline Model ---")
    base_model_path = f'stacking_model_{RANDOM_STATE}.pkl'
    
    if not os.path.exists(base_model_path):
        print(f"Error: Model file '{base_model_path}' not found.")
        print("Please run the base model training script first.")
        return None

    with open(base_model_path, 'rb') as f:
        base_model = pickle.load(f)

    # --- Load Dataset ---
    print(f"--- Loading Data from {DATASET_PATH} ---")
    try:
        data = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None

    # Separate Labeled and Unlabeled Data
    # Assuming -1 indicates unlabeled data in 'PatSat' column
    data_labeled = data[data['PatSat'] != -1].copy()
    
    X_labeled = data_labeled.drop('PatSat', axis=1)
    y_labeled = data_labeled['PatSat']
    
    num_labels = y_labeled.nunique()
    class_names = ["Neutral", "Positive", "Negative"] 

    # --- Preprocessing for Baseline (Strategy 1) ---
    model_smote = ADASYN(random_state=RANDOM_STATE) 
    X_train_resampled, y_train_resampled = model_smote.fit_resample(X_labeled, y_labeled)  
    # Split into Train and Test (Test set used for evaluation)
    print(f"--- Splitting Labeled Data (Train/Test) ---")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train_resampled, y_train_resampled, 
        test_size=TEST_SET_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y_train_resampled
    )

    print("--- Scaling Features (StandardScaler) ---")
    scaler = StandardScaler()
    scaler.fit(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================
    # Strategy 1: Baseline Evaluation
    # ==========================================
    print("\n" + "-"*50)
    print("--- Evaluating Strategy 1: Baseline (Pre-trained Stacking) ---")
    print("-"*50)
    
    run_results = {}
    
    y_pred_base = base_model.predict(X_test_scaled)
    report_base = classification_report(
        y_test, y_pred_base, target_names=class_names,
        labels=range(num_labels), digits=4, output_dict=True
    )
    run_results['Baseline'] = report_base
    print(f"[Baseline] Macro-F1: {report_base['macro avg']['f1-score']:.4f}")
    
    # Save ROC for Baseline
    plot_multiclass_roc(base_model, X_test_scaled, y_test, num_labels, class_names, 
                        f"roc_baseline_rs_{RANDOM_STATE}.tif")

    # ==========================================
    # Strategy 2: Self-Training (Pure)
    # ==========================================
    print("\n" + "-"*50)
    print("--- Evaluating Strategy 2: Self-Training (Standard) ---")
    print("-"*50)
    
    # Data Preparation for Semi-Supervised Learning
    # Combine all features and labels
    X_full = data.drop('PatSat', axis=1)
    y_full = data['PatSat']
    model_smote_ssl = ADASYN(random_state=RANDOM_STATE)
    X_smote_full, y_smote_full = model_smote_ssl.fit_resample(X_full, y_full)

    # Split data
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X_smote_full, y_smote_full, 
        test_size=TEST_SET_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y_smote_full
    )

    # Filter: Extract only valid labeled data for Testing
    # We reconstruct the dataframe to filter out any potentially remaining -1 labels in test set
    test_df = pd.concat([X_test_all, y_test_all], axis=1)
    test_df_valid = test_df[test_df['PatSat'] != -1]
    
    X_test_ssl = test_df_valid.drop('PatSat', axis=1)
    y_test_ssl = test_df_valid['PatSat']

    # Scaling for SSL
    scaler_ssl = StandardScaler()
    X_ssl_train_scaled = scaler_ssl.fit_transform(X_train_all)
    X_ssl_test_scaled = scaler_ssl.transform(X_test_ssl)

    print(f"Strategy 2 Training Set Size: {len(X_ssl_train_scaled)}")
    
    # Configuration for SelfTrainingClassifier
    ssl_params = {
        'threshold': 0.9,
        'criterion': 'threshold',
        'max_iter': 20,
        'verbose': True
    }

    # Train Self-Training Model
    # Note: y_train_all contains -1 (unlabeled), which SelfTrainingClassifier handles
    print("Training SelfTrainingClassifier...")
    ssl_model_pure = SelfTrainingClassifier(base_estimator=base_model, **ssl_params)
    ssl_model_pure.fit(X_ssl_train_scaled, y_train_all)
    
    # Evaluate Strategy 2
    y_pred_pure = ssl_model_pure.predict(X_ssl_test_scaled)
    report_pure = classification_report(
        y_test_ssl, y_pred_pure, target_names=class_names,
        labels=range(num_labels), digits=4, output_dict=True
    )
    run_results['Self-Training'] = report_pure
    print(f"[Self-Training] Macro-F1: {report_pure['macro avg']['f1-score']:.4f}")

    # Save Strategy 2 Model
    with open(f'ssl_model_rs_{RANDOM_STATE}.pkl', 'wb') as f:
        pickle.dump(ssl_model_pure, f)
        
    # Save ROC for Strategy 2
    plot_multiclass_roc(ssl_model_pure, X_ssl_test_scaled, y_test_ssl, num_labels, class_names, 
                        f"roc_ssl_rs_{RANDOM_STATE}.tif")

    print(f"--- Experiment for Seed {RANDOM_STATE} Completed ---")
    return run_results


# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    
    # Random seeds for reproducibility
    random_states_list = [100, 1000, 2025]
    all_run_results = []
    
    print(f"--- [Start] Starting experiments with {len(random_states_list)} random seeds ---")

    for i, rs in enumerate(random_states_list):
        print("\n" + "#"*70)
        print(f"--- Processing Run {i+1}/{len(random_states_list)} (Random State = {rs}) ---")
        print("#"*70)
        
        results = run_ssl_comparison(current_random_state=rs)
        
        if results is not None:
            all_run_results.append((rs, results))
        else:
            print(f"Skipping Random State {rs} due to errors.")
        
    print("\n" + "="*70)
    print("--- [Finish] All experiments completed. Aggregating results. ---")
    print("="*70)

    # --- Result Aggregation ---
    
    metrics_to_keep = ['precision', 'recall', 'f1-score']
    classes_to_keep = ["Neutral", "Positive", "Negative", "macro avg", "weighted avg"]
    
    records = []
    
    for rs, run_result in all_run_results:
        for model_name, report in run_result.items():
            if report is None:
                continue
            
            # Extract metrics per class and averages
            for class_name in classes_to_keep:
                if class_name in report:
                    record = {
                        'random_state': rs,
                        'model': model_name, # 'Baseline' or 'Self-Training'
                        'class': class_name
                    }
                    for metric in metrics_to_keep:
                        if metric in report[class_name]:
                            record[metric] = report[class_name][metric]
                    records.append(record)
            
            # Extract global accuracy
            if 'accuracy' in report:
                records.append({
                    'random_state': rs,
                    'model': model_name,
                    'class': 'accuracy',
                    'precision': report['accuracy'], # Placeholder in precision column
                    'recall': np.nan,
                    'f1-score': np.nan
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(records)
    
    if not results_df.empty:
        # Calculate Mean and Standard Deviation
        summary_mean = results_df.groupby(['model', 'class'])[metrics_to_keep].mean()
        summary_std = results_df.groupby(['model', 'class'])[metrics_to_keep].std()
        
        # Format as "Mean ± Std"
        flat_data = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_mean.values.flat, summary_std.values.flat)]
        reshaped_data = np.array(flat_data).reshape(summary_mean.shape)
        
        summary_combined = pd.DataFrame(
            reshaped_data,
            index=summary_mean.index,
            columns=summary_mean.columns
        ).unstack(level='class')
        
        print("\n--- Aggregated Results (Mean ± Std) ---")
        # Use try-except for markdown printing compatibility
        try:
            print(summary_combined.to_markdown())
        except AttributeError:
            print(summary_combined)

        # Save to Excel
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
                results_df.set_index(['random_state', 'model', 'class']).to_excel(writer, sheet_name='Raw_Data')
                summary_mean.to_excel(writer, sheet_name='Summary_Mean')
                summary_std.to_excel(writer, sheet_name='Summary_Std')
                summary_combined.to_excel(writer, sheet_name='Final_Report')
            
            print(f"\nSuccess: Full report saved to '{OUTPUT_EXCEL}'")
            
        except PermissionError:
            print(f"\n[Error] Could not save Excel file. Please close '{OUTPUT_EXCEL}' and try again.")
        except ImportError:
            print("\n[Error] 'openpyxl' library is missing. Install it via: pip install openpyxl")
    else:
        print("\n[Warning] No results were collected. Check input files and paths.")