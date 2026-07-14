import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_and_save_models():
    # 1. Load dataset
    csv_path = 'medical_equipment_data.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please place it in the root directory.")
    
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # 2. Encode categorical columns
    categorical_cols = ['equipment_type', 'manufacturer']
    label_encoders = {}
    
    print("Encoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}': {list(le.classes_)} -> {list(le.transform(le.classes_))}")
        
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    print("Saved label encoders to 'models/label_encoders.pkl'")
    
    # 3. Define features and targets
    feature_cols = [
        'equipment_type', 'manufacturer', 'age_months', 'usage_hours_per_day', 
        'total_scans_or_uses', 'days_since_last_maintenance', 'num_previous_repairs', 
        'avg_temperature_c', 'max_temperature_c', 'vibration_level_mm_s', 
        'voltage_fluctuation_pct', 'coolant_level_pct', 'helium_level_pct', 
        'tube_current_ma', 'error_logs_last_30_days', 'power_supply_stability_score', 
        'ambient_humidity_pct', 'component_wear_index', 'technician_rating_last_inspection'
    ]
    
    X = df[feature_cols]
    y_class = df['failure_within_30_days']
    y_reg = df['remaining_useful_life_days']
    
    print(f"Features for training ({len(feature_cols)}): {feature_cols}")
    
    # 4. Split data (80% train, 20% test)
    # Use different splits or same split? Let's use the same random state for splits to align features.
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # 5. Train Classification Models (Failure Probability within 30 days)
    print("\n--- Training Classification Models ---")
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss', n_jobs=-1)
    
    # Hybrid Voting Classifier (Soft Voting)
    hybrid_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft'
    )
    
    print("Fitting Hybrid Classifier...")
    hybrid_clf.fit(X_train_c, y_train_c)
    
    # Save hybrid classifier
    joblib.dump(hybrid_clf, 'models/failure_classifier.pkl')
    print("Saved Hybrid Classifier to 'models/failure_classifier.pkl'")
    
    # Evaluate Classifier
    y_pred_c = hybrid_clf.predict(X_test_c)
    y_pred_proba_c = hybrid_clf.predict_proba(X_test_c)[:, 1]
    
    acc = accuracy_score(y_test_c, y_pred_c)
    prec = precision_score(y_test_c, y_pred_c, zero_division=0)
    rec = recall_score(y_test_c, y_pred_c, zero_division=0)
    f1 = f1_score(y_test_c, y_pred_c, zero_division=0)
    roc_auc = roc_auc_score(y_test_c, y_pred_proba_c)
    cm = confusion_matrix(y_test_c, y_pred_c)
    
    print("\nClassifier Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save classification performance stats to a text file for model_info page
    with open('outputs/model_metrics.txt', 'w') as f:
        f.write("=== CLASSIFICATION METRICS (Hybrid Ensemble) ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC AUC:   {roc_auc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n\n")

    # 6. Train Regression Models (Remaining Useful Life in days)
    print("\n--- Training Regression Models ---")
    
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    
    # Hybrid Voting Regressor
    hybrid_reg = VotingRegressor(
        estimators=[('rf', rf_reg), ('xgb', xgb_reg)]
    )
    
    print("Fitting Hybrid Regressor...")
    hybrid_reg.fit(X_train_r, y_train_r)
    
    # Save hybrid regressor
    joblib.dump(hybrid_reg, 'models/rul_regressor.pkl')
    print("Saved Hybrid Regressor to 'models/rul_regressor.pkl'")
    
    # Evaluate Regressor
    y_pred_r = hybrid_reg.predict(X_test_r)
    
    mae = mean_absolute_error(y_test_r, y_pred_r)
    rmse = root_mean_squared_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    
    print("\nRegressor Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
    print(f"R² Score:                  {r2:.4f}")
    
    # Save regression performance stats to metrics file
    with open('outputs/model_metrics.txt', 'a') as f:
        f.write("=== REGRESSION METRICS (Hybrid Ensemble) ===\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f} days\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f} days\n")
        f.write(f"R2 Score:                  {r2:.4f}\n")
        
    print("\nModel training and validation complete!")

if __name__ == '__main__':
    train_and_save_models()
