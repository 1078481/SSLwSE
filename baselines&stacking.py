"""
Multi-model Comparative Analysis and Stacking Ensemble
-------------------------------------------------------------------------
Description:
    This script performs a comparative analysis of multiple machine learning 
    models and a Stacking Ensemble model. It includes data preprocessing,
    hyperparameter configuration, model training, 
    evaluation (Classification Report, ROC-AUC), and result visualization.

    The experiment is repeated over multiple random seeds to ensure statistical 
    robustness.

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

# --- Core Scikit-learn Libraries ---
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, roc_curve, auc, f1_score

# --- Imbalanced Learning ---
from imblearn.over_sampling import ADASYN

# --- Models ---
from sklearn.ensemble import StackingClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# --- Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Dataset Path (Adjust this path before running)
DATASET_PATH = 'dataset.xlsx' 
SHEET_NAME = 'all-2'

# --- 1. Visualization Configuration ---
try:
    config = {
        "font.family": 'serif',
        "font.size": 14,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman', 'DejaVu Serif'], # Standard academic fonts
    }
    plt.rcParams.update(config)
except Exception as e:
    print(f"Warning: Error setting font configurations. Using default fonts. Error: {e}")


# --- 2. Helper Functions ---

def plot_multiclass_roc(model, X_test, y_test, num_labels, class_names=None, filename="roc_curve.tif"):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for multi-class classification
    and saves the figure to a file.

    Args:
        model: Trained classifier with `predict_proba` method.
        X_test: Test features.
        y_test: True labels for the test set.
        num_labels (int): Number of unique classes.
        class_names (list, optional): List of class names for the legend.
        filename (str): Output filename for the plot.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_labels)]
    
    # Check if model supports probability prediction
    if not hasattr(model, "predict_proba"):
        print(f"Warning: Model {type(model).__name__} does not support 'predict_proba'. Skipping ROC plot.")
        return

    y_prob = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=list(range(num_labels)))
    
    if y_test_bin.shape[1] != y_prob.shape[1]:
        print(f"Error: Mismatch in number of classes. Labels: {y_test_bin.shape[1]}, Probs: {y_prob.shape[1]}")
        return

    colors = ['red', 'darkorange', 'cornflowerblue']
    linestyles = ["-", "--", ":"]

    plt.figure(figsize=(10, 8))
    
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        # Handle cases where color list is shorter than num_labels
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        
        plt.plot(fpr, tpr, color=color, linestyle=style, linewidth=3,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc:0.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Multi-class ROC Analysis', fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.savefig(filename, dpi=300, format="tif", transparent=True, pad_inches=0.1)
    print(f"ROC figure saved to {filename}")
    plt.close()


# --- 3. Main Training Pipeline ---

def run_experiment(current_random_state):
    """
    Executes the complete training and evaluation pipeline for a single random seed.

    Args:
        current_random_state (int): Seed for reproducibility.

    Returns:
        dict: A dictionary containing classification reports for all models.
    """
    
    # Constants
    RANDOM_STATE = current_random_state 
    TEST_SET_SIZE = 0.2
    
    print(f"\n--- Loading Data (Random State={RANDOM_STATE}) ---")
    try:
        data = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}. Please check the path.")
        return {}

    # Preprocessing: Filter valid data
    data_labeled = data[data['PatSat'] != -1].copy()
    X = data_labeled.drop('PatSat', axis=1)
    y = data_labeled['PatSat']
    
    num_labels = y.nunique()
    class_names = ["Neutral", "Positive", "Negative"] # Ensure these match your label encoding (0, 1, 2)
    
    print(f"Data loaded. Detected {num_labels} classes.")
    print(f"Original Class Distribution:\n{y.value_counts().sort_index()}")
    model_smote = ADASYN(random_state=RANDOM_STATE)
    X_resampled, y_resampled = model_smote.fit_resample(X, y)
    
    # Train/Test Split
    print("--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, 
        test_size=TEST_SET_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y_resampled
    )
    
    # Feature Scaling
    print("--- Scaling Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define Base Models
    print("--- Initializing Base Models ---")
    base_models = {
        'Logistic Regression': LogisticRegression(C=1000, max_iter=100, penalty='l2', random_state=RANDOM_STATE, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=30, random_state=RANDOM_STATE),
        'SVM': SVC(C=10, gamma=0.1, kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=2000, random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=154, random_state=RANDOM_STATE, n_jobs=-1),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    }

    # Train and Evaluate Base Models
    print("\n" + "="*50)
    print("--- Benchmarking Base Models ---")
    print("="*50)
    
    run_results = {}
    
    for name, model in base_models.items():
        print(f"Training: {name}...")
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            report_dict = classification_report(
                y_test, y_pred, 
                target_names=class_names, 
                labels=range(num_labels), 
                digits=4, output_dict=True
            )
            run_results[name] = report_dict
            
            print(f"[{name}] Accuracy: {report_dict['accuracy']:.4f}")
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            run_results[name] = None
    
    # --- Stacking Classifier ---
    print("\n" + "="*50)
    print("--- Training Stacking Ensemble ---")
    print("="*50)

    estimators_for_stacking = [
        ('knn', base_models['KNN']),
        ('svc', base_models['SVM']),
        ('xgb', base_models['XGBoost']),
        ('rf', base_models['Random Forest'])
    ]
    
    final_layer = LogisticRegression(random_state=RANDOM_STATE, max_iter=500, C=1.0, n_jobs=-1)
    
    stacking_model = StackingClassifier(
        estimators=estimators_for_stacking,
        final_estimator=final_layer,
        cv=5,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train_scaled, y_train)
    print("Stacking model training complete.")
    
    # Evaluate Stacking Model
    y_pred_stack = stacking_model.predict(X_test_scaled)
    report_dict_stack = classification_report(
        y_test, y_pred_stack, 
        target_names=class_names, 
        labels=range(num_labels), 
        digits=4, output_dict=True
    )
    run_results['Stacking'] = report_dict_stack
    
    print(f"\n[Stacking] Final Report:\n{classification_report(y_test, y_pred_stack, digits=4, target_names=class_names)}")

    # Save Model
    output_model_path = f'stacking_model_seed_{RANDOM_STATE}.pkl'
    with open(output_model_path, 'wb') as f:
        pickle.dump(stacking_model, f)
    print(f"Model saved to {output_model_path}")

    # Plot ROC for Stacking Model
    roc_fig_filename = f"stack_roc_seed_{RANDOM_STATE}.tif"
    plot_multiclass_roc(stacking_model, X_test_scaled, y_test, num_labels, class_names, roc_fig_filename)

    print(f"--- Experiment for Seed {RANDOM_STATE} Completed ---")
    return run_results


# --- 4. Main Execution Loop ---

if __name__ == "__main__":
    
    # List of random seeds for statistical validation
    random_states_list = [1, 10, 100, 1000, 2025]
    all_run_results = []
    
    print(f"--- [Start] Starting experiments with {len(random_states_list)} random seeds ---")

    for i, rs in enumerate(random_states_list):
        print(f"\nProcessing Run {i+1}/{len(random_states_list)} (Random State = {rs})")
        run_results_dict = run_experiment(current_random_state=rs)
        if run_results_dict:
            all_run_results.append(run_results_dict)
        
    print("\n" + "="*70)
    print("--- [Finish] All experiments completed. Aggregating results. ---")
    print("="*70)

    # --- Result Aggregation & Reporting ---
    
    metrics_to_keep = ['precision', 'recall', 'f1-score']
    # Define classes and average metrics to extract
    keys_to_extract = ["Neutral", "Positive", "Negative", "macro avg", "weighted avg"]
    
    records = []
    
    for i, run_result in enumerate(all_run_results):
        rs = random_states_list[i]
        for model_name, report in run_result.items():
            if report is None: 
                continue
            
            # Extract class-wise and average metrics
            for key in keys_to_extract:
                if key in report:
                    record = {
                        'random_state': rs,
                        'model': model_name,
                        'class_or_avg': key
                    }
                    for metric in metrics_to_keep:
                        record[metric] = report[key].get(metric, np.nan)
                    records.append(record)
            
            # Extract Global Accuracy
            if 'accuracy' in report:
                records.append({
                    'random_state': rs,
                    'model': model_name,
                    'class_or_avg': 'accuracy',
                    'precision': report['accuracy'], # Storing accuracy in precision col for simplification
                    'recall': np.nan,
                    'f1-score': np.nan
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(records)
    
    # Calculate Mean and Standard Deviation across runs
    summary_mean = results_df.groupby(['model', 'class_or_avg'])[metrics_to_keep].mean()
    summary_std = results_df.groupby(['model', 'class_or_avg'])[metrics_to_keep].std()
    
    # Format as "Mean ± Std"
    # Create a 2D array of formatted strings
    flat_data = [f"{m:.4f} ± {s:.4f}" for m, s in zip(summary_mean.values.flat, summary_std.values.flat)]
    reshaped_data = np.array(flat_data).reshape(summary_mean.shape)

    summary_combined = pd.DataFrame(
        reshaped_data,
        index=summary_mean.index,
        columns=summary_mean.columns
    ).unstack(level='class_or_avg')
    
    print("\n--- Aggregated Results (Mean ± Std over 5 runs) ---")
    # Using try-except for tabulate in case it's not installed, fallback to string
    try:
        print(summary_combined.to_markdown())
    except ImportError:
        print(summary_combined)

    # Save to Excel
    output_excel_path = "experiment_full_report.xlsx"
    try:
        with pd.ExcelWriter(output_excel_path) as writer:
            results_df.set_index(['random_state', 'model', 'class_or_avg']).to_excel(writer, sheet_name='Raw_Data')
            summary_mean.to_excel(writer, sheet_name='Summary_Mean')
            summary_std.to_excel(writer, sheet_name='Summary_Std')
            summary_combined.to_excel(writer, sheet_name='Final_Report')
            
        print(f"\nSuccess: Full report saved to '{output_excel_path}'")
        
    except PermissionError:
        print(f"\n[Error] Could not save Excel file. Please close '{output_excel_path}' and try again.")
    except Exception as e:
        print(f"\n[Error] Failed to save Excel: {e}")