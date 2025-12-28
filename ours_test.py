import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Configuration ---
MODEL_PATH = 'ssl_enhanced_model_rs_2025.pkl'
SCALER_PATH = 'scaler_enhanced_rs_2025.pkl'
DATASET_PATH = '3.xlsx'  # Using the dataset for "Ours"
SHEET_NAME = 'all-2'

# Text columns to drop (Model was trained on numerical features)
TEXT_COLUMNS = ['CasInf', 'DocMes', 'PatMes'] 

def evaluate_pretrained():
    print(f"--- Loading Pre-trained Resources ---")
    
    # 1. Load Model and Scaler
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Model and Scaler loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the current directory.")
        return

    # 2. Load Data
    print(f"--- Loading Test Data from {DATASET_PATH} ---")
    try:
        data = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 3. Preprocessing
    # Filter out unlabeled data (-1) to get the ground truth for evaluation
    data_labeled = data[data['PatSat'] != -1].copy()
    
    if len(data_labeled) == 0:
        print("Error: No labeled data found in the dataset for evaluation.")
        return

    # Separate Features and Target
    # Note: Strategy 3 model uses numerical features; text was used for pseudo-labeling only.
    X_test = data_labeled.drop(['PatSat'] + TEXT_COLUMNS, axis=1)
    y_test = data_labeled['PatSat']
    
    class_names = ["Neutral", "Positive", "Negative"]
    num_labels = len(class_names)

    # 4. Scaling
    # Crucial: Use the loaded scaler (do NOT fit a new one)
    X_test_scaled = scaler.transform(X_test)
    print(f"Evaluating on {len(X_test_scaled)} samples...")

    # 5. Prediction
    y_pred = model.predict(X_test_scaled)
    
    # 6. Report Generation
    print("\n" + "="*60)
    print(f"Performance Report for {MODEL_PATH}")
    print("="*60)
    
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 7. (Optional) Generate ROC Plot
    try:
        y_prob = model.predict_proba(X_test_scaled)
        y_test_bin = label_binarize(y_test, classes=list(range(num_labels)))
        
        plt.figure(figsize=(8, 6))
        colors = ['red', 'darkorange', 'cornflowerblue']
        for i in range(num_labels):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Pre-trained Model)')
        plt.legend(loc="lower right")
        plt.savefig("pretrained_model_roc.tif", dpi=300)
        print("\nROC Curve saved to 'pretrained_model_roc.tif'")
    except AttributeError:
        print("Model does not support probability output, skipping ROC.")

if __name__ == "__main__":
    evaluate_pretrained()