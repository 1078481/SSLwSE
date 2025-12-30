import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- System Configuration ---
# Inference Resources
MODEL_PATH = 'ssl_enhanced_model_rs_2025.pkl'
SCALER_PATH = 'scaler_enhanced_rs_2025.pkl'

# Target Data for Verification
DATASET_PATH = '3.xlsx'  
SHEET_NAME = 'all-2'

# Output for analysis
OUTPUT_RESULT_FILE = 'verification_results.csv'

# Feature Configuration
# Columns that are text-based and need to be excluded from the numerical model
TEXT_COLUMNS = ['CasInf', 'DocMes', 'PatMes'] 

def run_inference_verification():
    """
    Loads the pre-trained production model and performs inference on the target dataset
    to verify actual performance metrics.
    """
    print(f"--- Initializing Inference Environment ---")
    
    # 1. Load Production Resources (Model & Scaler)
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[OK] Model loaded: {MODEL_PATH}")
        print(f"[OK] Scaler loaded: {SCALER_PATH}")
    except FileNotFoundError as e:
        print(f"[Error] Resource missing: {e}")
        return

    # 2. Load Target Data
    print(f"--- Loading Target Data from {DATASET_PATH} ---")
    try:
        data = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"[Error] Failed to read data source: {e}")
        return

    # 3. Data Preprocessing & Validation Setup
    # Filter valid data for verification (PatSat != -1)
    # Note: In a pure inference scenario without ground truth, we would skip this filter.
    target_data = data[data['PatSat'] != -1].copy()
    
    if len(target_data) == 0:
        print("[Warning] No valid records found for verification.")
        return

    # Prepare Input Features (X) and Ground Truth (y)
    X_inference = target_data.drop(['PatSat'] + TEXT_COLUMNS, axis=1)
    y_ground_truth = target_data['PatSat']
    
    class_names = ["Neutral", "Positive", "Negative"]
    num_labels = len(class_names)

    # 4. Feature Scaling (Normalization)
    # Strictly apply the training scaler to the new data
    X_inference_scaled = scaler.transform(X_inference)
    print(f"-> Ready to infer on {len(X_inference_scaled)} instances.")

    # 5. Execute Inference
    print(f"--- Executing Model Inference ---")
    y_pred = model.predict(X_inference_scaled)
    
    # Try getting probabilities for detailed analysis
    try:
        y_prob = model.predict_proba(X_inference_scaled)
    except AttributeError:
        y_prob = None
        print("[Info] Model does not provide probability estimates.")

    # 6. Save Inference Results (New Step)
    # Save the predictions alongside actuals for manual inspection
    results_df = X_inference.copy()
    results_df['Actual_Status'] = y_ground_truth
    results_df['Predicted_Status'] = y_pred
    if y_prob is not None:
        for i, name in enumerate(class_names):
            results_df[f'Prob_{name}'] = y_prob[:, i]
            
    results_df.to_csv(OUTPUT_RESULT_FILE, index=False)
    print(f"[Output] Detailed inference results saved to '{OUTPUT_RESULT_FILE}'")

    # 7. Verification Report
    print("\n" + "="*60)
    print(f"VERIFICATION REPORT: Strategy 3 (Ours)")
    print("="*60)
    
    print(classification_report(y_ground_truth, y_pred, target_names=class_names, digits=4))
    acc = accuracy_score(y_ground_truth, y_pred)
    print(f"Verification Accuracy: {acc:.4f}")

    # 8. Visual Validation (ROC Curve)
    if y_prob is not None:
        generate_validation_plots(y_ground_truth, y_prob, num_labels, class_names)

def generate_validation_plots(y_true, y_prob, num_labels, class_names):
    """Generates visual artifacts to validate model behavior."""
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_labels)))
        
        plt.figure(figsize=(8, 6))
        colors = ['red', 'darkorange', 'cornflowerblue']
        
        for i in range(num_labels):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Verification ROC Curve (Strategy 3)')
        plt.legend(loc="lower right")
        
        output_plot = "verification_roc.tif"
        plt.savefig(output_plot, dpi=300)
        print(f"[Output] ROC plot saved to '{output_plot}'")
        
    except Exception as e:
        print(f"[Warning] Could not generate plots: {e}")

if __name__ == "__main__":
    run_inference_verification()