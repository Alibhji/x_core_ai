import numpy as np
import torch
from typing import Dict, List, Union, Tuple, Optional, Any

class Luid:
    """
    LUID (Local Unified Interpretability Decoder) for model interpretability.
    
    This class provides methods for analyzing and explaining model predictions
    through various techniques for feature importance and model behavior analysis.
    """
    
    def __init__(self, model, config=None):
        """
        Initialize the Luid interpretability module.
        
        Args:
            model: The trained model to interpret
            config: Configuration dictionary with parameters for interpretability
        """
        self.model = model
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def analyze_feature_importance(self, inputs, targets=None, method="shap"):
        """
        Analyze feature importance for model predictions.
        
        Args:
            inputs: Input data for which to analyze feature importance
            targets: Optional target values for reference
            method: Method to use for importance analysis ('shap', 'lime', 'integrated_gradients')
            
        Returns:
            Dictionary containing feature importance scores
        """
        if method == "shap":
            return self._compute_shap_values(inputs)
        elif method == "lime":
            return self._compute_lime_importance(inputs)
        elif method == "integrated_gradients":
            return self._compute_integrated_gradients(inputs)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _compute_shap_values(self, inputs):
        """
        Compute SHAP values for feature importance.
        
        Args:
            inputs: Input data to analyze
            
        Returns:
            Dictionary with SHAP values
        """
        # Placeholder for actual SHAP implementation
        # Would typically use shap library
        print("Computing SHAP values")
        
        # Dummy implementation
        if isinstance(inputs, torch.Tensor):
            importance = torch.rand_like(inputs)
        else:
            importance = np.random.rand(*inputs.shape)
            
        return {"shap_values": importance}
    
    def _compute_lime_importance(self, inputs):
        """
        Compute LIME importance scores.
        
        Args:
            inputs: Input data to analyze
            
        Returns:
            Dictionary with LIME importance values
        """
        # Placeholder for actual LIME implementation
        print("Computing LIME importance")
        
        # Dummy implementation
        if isinstance(inputs, torch.Tensor):
            importance = torch.rand_like(inputs)
        else:
            importance = np.random.rand(*inputs.shape)
            
        return {"lime_importance": importance}
    
    def _compute_integrated_gradients(self, inputs):
        """
        Compute Integrated Gradients for feature importance.
        
        Args:
            inputs: Input data to analyze
            
        Returns:
            Dictionary with integrated gradients values
        """
        # Placeholder for actual Integrated Gradients implementation
        print("Computing Integrated Gradients")
        
        # Dummy implementation
        if isinstance(inputs, torch.Tensor):
            importance = torch.rand_like(inputs)
        else:
            importance = np.random.rand(*inputs.shape)
            
        return {"integrated_gradients": importance}
    
    def visualize_importance(self, importance_scores, feature_names=None, top_k=10):
        """
        Visualize feature importance scores.
        
        Args:
            importance_scores: Dictionary with importance scores from analyze_feature_importance
            feature_names: Optional list of feature names
            top_k: Number of top features to display
            
        Returns:
            Dictionary with visualization data
        """
        import matplotlib.pyplot as plt
        
        # Extract scores from the first method found in the dictionary
        method_name = list(importance_scores.keys())[0]
        scores = importance_scores[method_name]
        
        # Handle different input shapes
        if len(scores.shape) > 2:
            # For higher dimensional inputs, flatten or take mean
            if isinstance(scores, torch.Tensor):
                flat_scores = scores.abs().mean(dim=0).flatten().cpu().numpy()
            else:
                flat_scores = np.abs(scores).mean(axis=0).flatten()
        else:
            if isinstance(scores, torch.Tensor):
                flat_scores = scores.abs().flatten().cpu().numpy()
            else:
                flat_scores = np.abs(scores).flatten()
        
        # Sort and get top_k features
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(flat_scores))]
        
        # Ensure feature_names matches the length of flat_scores
        if len(feature_names) != len(flat_scores):
            feature_names = [f"Feature {i}" for i in range(len(flat_scores))]
            
        # Get indices of top features
        top_indices = np.argsort(flat_scores)[-top_k:]
        top_scores = flat_scores[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_scores, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Important Features ({method_name})')
        plt.tight_layout()
        
        # Save visualization
        output_path = "feature_importance.png"
        plt.savefig(output_path)
        plt.close()
        
        return {
            "visualization_path": output_path,
            "top_features": top_features,
            "top_scores": top_scores.tolist() if hasattr(top_scores, 'tolist') else top_scores
        }
    
    def compare_models(self, models, inputs, targets=None):
        """
        Compare interpretability results across multiple models.
        
        Args:
            models: List of models to compare
            inputs: Input data for evaluation
            targets: Optional target values
            
        Returns:
            Comparison results
        """
        results = []
        
        # Store current model
        original_model = self.model
        
        try:
            # Analyze each model
            for i, model in enumerate(models):
                self.model = model
                importance = self.analyze_feature_importance(inputs)
                results.append({
                    "model_index": i,
                    "importance": importance
                })
        finally:
            # Restore original model
            self.model = original_model
            
        return results 