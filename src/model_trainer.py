#lightgbm code src/model_trainer.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data_path='data/training_data_4.csv', models_dir='models'):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def load_and_split_data(self, target_col):
        """
        Loads data and splits strictly by TIME.
        Train = Jan 2024 to Oct 2025
        Val = Oct 2025 to Jan 2026 (The Future)
        """
        print(f"⏳ Loading {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Define Split Date (The 'Future' Boundary)
        split_date = pd.Timestamp("2025-10-01").tz_localize('UTC')
        
        print(f"✂️  Splitting data at {split_date} (Train vs Future)...")

        # Identify features (All columns except time, raw prices, and targets)
        drop_cols = ['time', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<SPREAD>', '<VOL>', 
                     'target_1m_call', 'target_1m_put',
                     'target_2m_call', 'target_2m_put',
                     'target_3m_call', 'target_3m_put',
                     'target_4m_call', 'target_4m_put',
                     'target_5m_call', 'target_5m_put']
        
        # Keep features only
        feature_cols = [c for c in df.columns if c not in drop_cols]
        print(f"    Features detected: {len(feature_cols)}")

        # Perform the Split
        train_df = df[df['time'] < split_date]
        val_df = df[df['time'] >= split_date]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        # Identify Categorical Features (Asset ID)
        cat_features = ['asset_id'] 
        # Note: CatBoost expects indices or names. If asset_id is a string, we are good.
        
        print(f"    Training Rows:   {len(X_train)}")
        print(f"    Validation Rows: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val, cat_features

    def train(self, target_col='target_3m_call'):
        """Trains the CatBoost Model"""
        X_train, y_train, X_val, y_val, cat_features = self.load_and_split_data(target_col)
        
        # Initialize CatBoost
        # We use a balanced setup for speed and accuracy
        model = CatBoostClassifier(
            iterations=3000,           # Max trees
            learning_rate=0.03,        # Step size
            depth=8,                   # Tree depth (6 is standard)
            loss_function='Logloss',   # Binary classification
            eval_metric='Accuracy',    # What to optimize
            early_stopping_rounds=100,  # Stop if no improvement
            verbose=100,               # Print every 100 steps
            cat_features=cat_features, # Handle Asset IDs
            task_type="CPU"            # Change to GPU if you have NVIDIA setup
        )

        print(f"\n🔥 Ignition! Training model for {target_col}...")
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        # Evaluate
        print("\n📊 Evaluation on Unseen Future Data:")
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1] # Probability of Class 1
        
        acc = accuracy_score(y_val, preds)
        print(f"    -> Raw Accuracy (0.5 threshold): {acc:.4f}")
        
        # NEW V2 "BILLIONAIRE" DASHBOARD
        print("\n--- Confidence-Filtered Performance ---")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            high_conf_mask = probs > threshold
            trade_count = sum(high_conf_mask)
            
            if trade_count > 0:
                high_conf_accuracy = accuracy_score(y_val[high_conf_mask], preds[high_conf_mask])
                print(f"    -> Win Rate at >{int(threshold*100)}% Confidence: {high_conf_accuracy:.4f} ({trade_count} trades)")
            else:
                print(f"    -> No trades found with >{int(threshold*100)}% confidence.")

        # Save
        model_name = f"catboost_{target_col}.cbm"
        save_path = os.path.join(self.models_dir, model_name)
        model.save_model(save_path)
        print(f"\n💾 Model saved to: {save_path}")
        
        # Feature Importance Plot
        self.plot_importance(model, X_train.columns)

    def plot_importance(self, model, feature_names):
        """Generates the Audit Report"""
        importance = model.get_feature_importance()
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(40)
        
        print("\n🔍 TOP 20 ALPHA DRIVERS:")
        print(fi_df)
        
        # Optional: Save plot
        plt.figure(figsize=(10, 8))
        plt.barh(fi_df['feature'], fi_df['importance'])
        plt.gca().invert_yaxis()
        plt.title('Alpha Factory: Feature Importance')
        plt.savefig('feature_importance.png')
        print("    -> Chart saved as feature_importance.png")

if __name__ == "__main__":
    # You can change the target here to test differenct expiries
    trainer = ModelTrainer(data_path='data/training_data_4.csv')
    trainer.train(target_col='target_3m_call')