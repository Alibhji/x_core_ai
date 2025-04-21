# LUID: Local Unified Interpretability Decoder

LUID is a comprehensive tool for model interpretability that unifies multiple feature importance methods to help explain model predictions.

## Overview

LUID (Local Unified Interpretability Decoder) provides:

1. A unified interface for different interpretability methods
2. Feature importance analysis for model predictions
3. Visualization tools for understanding model behavior
4. Comparison capabilities across multiple models

## Supported Methods

LUID currently supports these interpretability techniques:

### SHAP (SHapley Additive exPlanations)

SHAP assigns each feature an importance value for a particular prediction based on game theory concepts. It provides consistent and locally accurate attributions by examining all possible feature subsets.

```python
# Get SHAP importance values
shap_importance = luid.analyze_feature_importance(input_data, method="shap")
```

### LIME (Local Interpretable Model-agnostic Explanations)

LIME explains model predictions by approximating the model locally with an interpretable model. It perturbs the input and observes changes to learn which features are most influential.

```python
# Get LIME importance values
lime_importance = luid.analyze_feature_importance(input_data, method="lime")
```

### Integrated Gradients

Integrated Gradients attributes a model's prediction to its input features by computing the gradients of the output with respect to input features along a straight-line path from a baseline to the input.

```python
# Get Integrated Gradients importance values
ig_importance = luid.analyze_feature_importance(input_data, method="integrated_gradients")
```

## Usage Examples

### Basic Usage

```python
import torch
from src.core import Core
from src.interpretability.luid.luid import Luid

# Initialize with a model
core = Core(config)
core.model_generator()
luid = Luid(core.model, config)

# Create input data
input_data = torch.randn(1, 10, 64)  # Example shape: (batch, sequence, features)

# Get feature importance
importance = luid.analyze_feature_importance(input_data, method="shap")

# Define feature names (optional)
feature_names = [f"Feature_{i}" for i in range(64)]

# Visualize importance
viz_result = luid.visualize_importance(
    importance, 
    feature_names=feature_names,
    top_k=10
)

# Display results
print(f"Visualization saved to: {viz_result['visualization_path']}")
print("Top features:")
for feature, score in zip(viz_result['top_features'], viz_result['top_scores']):
    print(f"  {feature}: {score:.4f}")
```

### Comparing Models

```python
# Create a second model for comparison
second_model = create_another_model()

# Compare interpretability across models
results = luid.compare_models(
    [core.model, second_model],
    input_data
)

# Analyze differences in explanations
for i, result in enumerate(results):
    print(f"Model {i+1} top features:")
    importance = result["importance"]
    viz = luid.visualize_importance(importance)
    print(f"  Visualization: {viz['visualization_path']}")
```

## Interactive UI

LUID comes with an interactive web UI for exploring model interpretability, built with Streamlit. To launch the UI:

```bash
python demo/luid_demo.py --open-ui
```

The UI provides:
- Method selection (SHAP, LIME, Integrated Gradients)
- Feature importance visualization
- Ability to generate new random inputs
- Model information display
- Educational explanations of each method

## API Reference

### Luid Class

```python
Luid(model, config=None)
```

**Parameters:**
- `model`: The trained model to interpret
- `config`: Configuration dictionary with parameters for interpretability

### Methods

#### analyze_feature_importance

```python
analyze_feature_importance(inputs, targets=None, method="shap")
```

**Parameters:**
- `inputs`: Input data for which to analyze feature importance
- `targets`: Optional target values for reference
- `method`: Method to use ('shap', 'lime', 'integrated_gradients')

**Returns:**
- Dictionary containing feature importance scores

#### visualize_importance

```python
visualize_importance(importance_scores, feature_names=None, top_k=10)
```

**Parameters:**
- `importance_scores`: Dictionary with importance scores
- `feature_names`: Optional list of feature names
- `top_k`: Number of top features to display

**Returns:**
- Dictionary with visualization data including path, top features, and scores

#### compare_models

```python
compare_models(models, inputs, targets=None)
```

**Parameters:**
- `models`: List of models to compare
- `inputs`: Input data for evaluation
- `targets`: Optional target values

**Returns:**
- List of comparison results for each model

## Implementation Details

The current implementation provides placeholder functions for each method, which will be replaced with actual implementations using libraries like:

- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
- [Captum](https://captum.ai/) (for Integrated Gradients)

## Future Enhancements

Planned enhancements for LUID:

1. Full implementations of all methods using their respective libraries
2. Support for more model types (CNNs, Transformers)
3. Additional visualization options
4. Counterfactual explanations
5. Global model interpretability
6. Export/import of explanations
7. Batch processing for large datasets 