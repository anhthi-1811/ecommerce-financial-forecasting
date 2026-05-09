import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Tắt các cảnh báo lặt vặt của SHAP
warnings.filterwarnings("ignore")

class ModelExplainer:
    """
    A class to support Machine Learning model explainability using SHAP and Feature Importance.
    Highly optimized for Tree-based models (LightGBM, CatBoost, XGBoost).
    """
    def __init__(self, model, X_data, model_name="CatBoost", output_dir="assets/"):
        self.model = model
        self.X_data = X_data.copy()
        self.model_name = model_name
        self.output_dir = output_dir 
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # ====================================================================
        # BYPASS CATEGORICAL ERRORS FOR SHAP
        # Convert categories to numeric codes and pass as a pure NumPy array.
        # This completely blinds LightGBM/XGBoost from Pandas metadata checks!
        # ====================================================================
        X_shap_safe = self.X_data.copy()
        for col in X_shap_safe.select_dtypes(include=['category', 'object']).columns:
            # Convert categories to underlying integer codes
            X_shap_safe[col] = X_shap_safe[col].cat.codes.astype(float)
            
        # Compute SHAP values using the pure NumPy array (.values)
        self.shap_values = self.explainer.shap_values(X_shap_safe.values)

    def plot_feature_importance(self, top_n=15):
        """
        Plots the default Feature Importance of the model.
        """
        plt.figure(figsize=(10, 6))
        
        # Extract feature importance based on the model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_feature_importance'): # For CatBoost
            importances = self.model.get_feature_importance()
        else:
            print("Model does not support default feature_importances_.")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        features = self.X_data.columns[indices]
        
        sns.barplot(x=importances[indices], y=features, palette="viridis")
        plt.title(f"Top {top_n} Feature Importance - {self.model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f"{self.model_name}_feature_importance.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Feature Importance plot saved successfully at: {save_path}")

    def plot_shap_summary(self):
        """
        Plots the SHAP Summary Plot (density and impact of features on predictions).
        """
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Summary Plot - {self.model_name}", fontsize=14, fontweight='bold')
        
        # Pass the original X_data so SHAP can display the original feature names!
        shap.summary_plot(self.shap_values, self.X_data, show=False)
        
        save_path = os.path.join(self.output_dir, f"{self.model_name}_shap_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"SHAP Summary plot saved successfully at: {save_path}")

    def plot_shap_dependence(self, feature_name):
        """
        Plots the SHAP Dependence Plot (similar to Partial Dependence Plot).
        Shows how the value of a specific feature affects the prediction.
        """
        if feature_name not in self.X_data.columns:
            print(f"Error: Feature '{feature_name}' not found in the dataset.")
            return
            
        plt.figure(figsize=(8, 5))
        plt.title(f"SHAP Dependence Plot: {feature_name} - {self.model_name}", fontsize=12, fontweight='bold')
        
        shap.dependence_plot(feature_name, self.shap_values, self.X_data, show=False)
        
        save_path = os.path.join(self.output_dir, f"{self.model_name}_shap_dep_{feature_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"SHAP Dependence plot saved successfully at: {save_path}")