# Model Interpretability

This package provides tools for understanding and explaining model predictions in the X-Core AI framework.

## Overview

Model interpretability is essential for:
- Understanding how models make decisions
- Building trust in AI systems
- Identifying potential biases
- Debugging model behavior
- Meeting regulatory requirements

This package includes various approaches to model interpretability, focusing on different aspects of model behavior and prediction explanation.

## Modules

### LUID (Local Unified Interpretability Decoder)

The LUID module provides a unified approach to model interpretability by supporting multiple explanation methods:

- **SHAP (SHapley Additive exPlanations)**: Explains the output of any model by computing Shapley values.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains predictions by approximating the model locally.
- **Integrated Gradients**: Attributes predictions to input features by computing gradients along a path.

## Getting Started

### Basic Usage

```python
from src.core import Core
from src.interpretability.luid.luid import Luid

# Initialize model
core = Core(config)
core.model_generator()

# Create LUID interpreter
luid = Luid(core.model, config)

# Generate input data
input_data = torch.randn(1, sequence_length, feature_dim)

# Analyze feature importance using SHAP
shap_importance = luid.analyze_feature_importance(input_data, method="shap")

# Visualize the results
feature_names = [f"Feature_{i}" for i in range(feature_dim)]
vis_result = luid.visualize_importance(shap_importance, feature_names=feature_names)
```

### Interactive UI

The package includes an interactive web UI for exploring model interpretability:

```bash
# Run the demo and open the UI
python demo/luid_demo.py --open-ui
```

The UI allows you to:
- Select different interpretation methods
- Adjust visualization parameters
- Generate new random inputs
- Compare feature importance across methods

## Configuration

You can configure the interpretability tools by adding parameters to your model configuration:

```python
config = {
    # ... existing model config ...
    
    "interpretability": {
        "default_method": "shap",
        "top_k_features": 10,
        "visualization_dir": "visualizations"
    }
}
```

## Extending the Framework

To add new interpretability methods:

1. Extend the `Luid` class with a new method implementation
2. Add the method to the supported methods in `analyze_feature_importance`
3. Implement the visualization handling for the new method

Example:

```python
def _compute_custom_importance(self, inputs):
    """
    Compute custom importance scores.
    
    Args:
        inputs: Input data to analyze
        
    Returns:
        Dictionary with custom importance values
    """
    # Custom implementation here
    
    return {"custom_importance": importance_scores}
```

## Future Developments

Planned enhancements for the interpretability package:

- Support for text and image models
- Additional interpretability methods
- Interactive counterfactual explanations
- Fairness analysis tools
- Integration with model monitoring 