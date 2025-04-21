import sys
import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import our modules
paths = [r'C:\Users\alibh\Desktop\projects\python', r'C:\Users\alibh\Desktop\projects\python\x_core_ai']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import our modules
from src.core import Core
from src.interpretability.luid.luid import Luid
from sub_module.utilx.src.config import ConfigLoader

def create_dummy_config():
    """Create a dummy configuration for the demo"""
    config = {
        "project_name": "luid_demo",
        "version": "v1.0",
        "model_name": "mlp",
        "model_kwargs": {
            "input_dim": 64,
            "hidden_dims": [128, 64],
            "output_dim": 1,
            "activation": "relu",
            "dropout": 0.2
        },
        "dataframe_kwargs": {
            "photo_feature_sequence_length": 10,
            "photo_feature_dim": 64,
            "data_split": {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.1
            }
        },
        "dataset_name": "dummy_dataset",
        "dataset_kwargs": {
            "num_samples": 100,
            "feature_dim": 64,
            "sequence_length": 10,
            "random_seed": 42
        },
        "dataloader_kwargs_train": {
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 0
        },
        "dataloader_kwargs": {
            "batch_size": 16,
            "shuffle": False,
            "num_workers": 0
        },
        "target_name": "cost_target",
        "metrics": ["mse", "mae", "r2"],
        "loss": "mse",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "epochs": 5
    }
    return config

def demo_luid(config):
    """Demonstrate Luid functionality"""
    print("\n===== Luid Interpretability Demo =====")
    
    # Initialize Core class and generate model
    core = Core(config)
    core.model_generator()
    print(f"Created model: {core.model.__class__.__name__}")
    
    # Initialize Luid interpreter
    luid = Luid(core.model, config)
    print("Initialized Luid interpreter")
    
    # Create dummy input
    dummy_input = torch.randn(1, 
                            config['dataframe_kwargs']['photo_feature_sequence_length'],
                            config['dataframe_kwargs']['photo_feature_dim'])
    print(f"Created dummy input with shape: {dummy_input.shape}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance using different methods:")
    
    # Using SHAP
    print("\n1. Using SHAP method:")
    shap_importance = luid.analyze_feature_importance(dummy_input, method="shap")
    
    # Using LIME
    print("\n2. Using LIME method:")
    lime_importance = luid.analyze_feature_importance(dummy_input, method="lime")
    
    # Using Integrated Gradients
    print("\n3. Using Integrated Gradients method:")
    ig_importance = luid.analyze_feature_importance(dummy_input, method="integrated_gradients")
    
    # Visualize importance scores
    print("\nVisualizing feature importance:")
    feature_names = [f"Feature_{i}" for i in range(config['dataframe_kwargs']['photo_feature_dim'])]
    vis_result = luid.visualize_importance(shap_importance, feature_names=feature_names, top_k=10)
    print(f"Visualization saved to: {vis_result['visualization_path']}")
    
    # Display top features
    print("\nTop important features:")
    for i, (feature, score) in enumerate(zip(vis_result['top_features'], vis_result['top_scores'])):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Compare with a second model (for demo purposes, just using the same model)
    print("\nComparing feature importance across models:")
    comparison = luid.compare_models([core.model, core.model], dummy_input)
    print(f"Compared {len(comparison)} models")
    
    return luid, core, dummy_input, feature_names

def open_luid_ui(luid=None, model=None, sample_input=None, feature_names=None):
    """
    Open a web UI for interacting with the Luid interpretability package
    
    Args:
        luid: Luid instance (will be created if None)
        model: Model instance (will be created if None)
        sample_input: Sample input data (will be created if None)
        feature_names: Feature names (will be created if None)
    """
    # Import required modules
    import subprocess
    import tempfile
    import webbrowser
    
    try:
        import streamlit
        has_streamlit = True
    except ImportError:
        has_streamlit = False
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        import streamlit
    
    # Create app.py in a temporary directory
    temp_dir = tempfile.mkdtemp()
    app_path = os.path.join(temp_dir, "luid_app.py")
    
    # Define the Streamlit app code
    app_code = '''
import streamlit as st
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Add paths to import modules
paths = [r'{root_path}', r'{parent_path}']
for path in paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Override the module structure to avoid x_core_ai import issues
import src
import src.core.core_base
import src.interpretability.luid.luid

# Import core classes directly
from src.core.core_base import Core
from src.interpretability.luid.luid import Luid

# Set page config
st.set_page_config(
    page_title="Luid Interpretability UI",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Luid Model Interpretability UI")
st.markdown("""
This interface allows you to interact with the Luid interpretability tools to:
- Analyze feature importance using different methods
- Visualize the most important features
- Compare interpretability across models
""")

# Helper function to get cached model
@st.cache_resource
def get_model():
    """Get model from cache or create new one"""
    config = {{
        "project_name": "luid_demo",
        "version": "v1.0",
        "model_name": "mlp",
        "model_kwargs": {{
            "input_dim": 64,
            "hidden_dims": [128, 64],
            "output_dim": 1,
            "activation": "relu", 
            "dropout": 0.2
        }},
        "dataframe_kwargs": {{
            "photo_feature_sequence_length": 10,
            "photo_feature_dim": 64
        }},
        # Add package_name to avoid x_core_ai import
        "package_name": "src"
    }}
    
    core = Core(config)
    core.model_generator()
    return core, config

# Get or create model
core, config = get_model()
luid = Luid(core.model, config)

# Create default input if needed
def get_sample_input():
    return torch.randn(1, 
                       config['dataframe_kwargs']['photo_feature_sequence_length'],
                       config['dataframe_kwargs']['photo_feature_dim'])

# Sidebar controls
st.sidebar.header("Analysis Options")
method = st.sidebar.selectbox(
    "Interpretation Method",
    ["shap", "lime", "integrated_gradients"],
    format_func=lambda x: {{"shap": "SHAP", "lime": "LIME", "integrated_gradients": "Integrated Gradients"}}[x]
)

top_k = st.sidebar.slider("Number of top features", 5, 20, 10)

st.sidebar.header("Input Options")
input_option = st.sidebar.radio(
    "Input Data",
    ["Use Default", "Generate New Random"]
)

# Generate input based on selection
if input_option == "Generate New Random":
    sample_input = get_sample_input()
    st.sidebar.success("Generated new random input")
else:
    sample_input = get_sample_input()

# Generate feature names
feature_names = [f"Feature_{{i}}" for i in range(config['dataframe_kwargs']['photo_feature_dim'])]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Feature Importance Analysis")
    
    if st.button(f"Run Analysis with {{method.upper()}}"):
        with st.spinner(f"Running {{method.upper()}} analysis..."):
            # Get importance scores
            importance = luid.analyze_feature_importance(sample_input, method=method)
            
            # Visualize
            st.subheader("Feature Importance Visualization")
            
            # Create a matplotlib figure
            fig = plt.figure(figsize=(10, 6))
            
            # Extract scores from the method
            method_key = list(importance.keys())[0]
            scores = importance[method_key]
            
            # Handle different input shapes
            if len(scores.shape) > 2:
                if isinstance(scores, torch.Tensor):
                    flat_scores = scores.abs().mean(dim=0).flatten().cpu().numpy()
                else:
                    flat_scores = np.abs(scores).mean(axis=0).flatten()
            else:
                if isinstance(scores, torch.Tensor):
                    flat_scores = scores.abs().flatten().cpu().numpy()
                else:
                    flat_scores = np.abs(scores).flatten()
            
            # Ensure feature_names are correct length
            if len(feature_names) != len(flat_scores):
                feature_names = [f"Feature_{{i}}" for i in range(len(flat_scores))]
            
            # Get indices of top features
            top_indices = np.argsort(flat_scores)[-top_k:][::-1]  # Reverse to show highest first
            
            # Ensure indices are within range
            top_indices = [i for i in top_indices if i < len(feature_names)]
            top_scores = flat_scores[top_indices]
            top_features = [feature_names[i] for i in top_indices]
            
            # Plot
            plt.barh(range(len(top_features)), top_scores, align='center')
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Importance Score')
            plt.title(f'Top {{top_k}} Important Features ({{method.upper()}})')
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            
            # Show data table of results
            st.subheader("Feature Importance Scores")
            importance_df = {{
                "Feature": top_features,
                "Importance Score": top_scores
            }}
            st.dataframe(importance_df)

with col2:
    st.header("Model Information")
    st.write(f"**Model Type:** {{core.model.__class__.__name__}}")
    
    # Model summary - Simple text representation
    model_summary = []
    model_summary.append(f"**Input Dimension:** {{config['model_kwargs']['input_dim']}}")
    model_summary.append(f"**Hidden Layers:** {{config['model_kwargs']['hidden_dims']}}")
    model_summary.append(f"**Output Dimension:** {{config['model_kwargs']['output_dim']}}")
    model_summary.append(f"**Activation:** {{config['model_kwargs']['activation']}}")
    
    for line in model_summary:
        st.write(line)
    
    # Input data information
    st.header("Input Data")
    st.write(f"**Input Shape:** {{list(sample_input.shape)}}")
    
    # Additional explanation
    st.header("About the Methods")
    method_explanation = {{
        "shap": """
        **SHAP (SHapley Additive exPlanations)**
        
        SHAP values explain how much each feature contributes to the prediction,
        based on game theory principles. Higher values indicate stronger influence
        on the model output.
        """,
        
        "lime": """
        **LIME (Local Interpretable Model-agnostic Explanations)**
        
        LIME explains predictions by approximating the model locally with
        an interpretable model. It perturbs the input and observes changes
        to understand feature importance.
        """,
        
        "integrated_gradients": """
        **Integrated Gradients**
        
        Integrated Gradients calculates feature importance by integrating
        gradients along a path from a baseline to the input. It attributes
        the prediction to input features based on the path integral.
        """
    }}
    
    st.markdown(method_explanation[method])

st.sidebar.header("About")
st.sidebar.markdown("""
**X-Core AI Luid Interpretability Package**

A tool for model interpretability and feature importance analysis.

Version: 1.0
""")
'''.format(
    root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    parent_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
    
    # Write the app code to a temporary file - use UTF-8 encoding explicitly
    with open(app_path, 'w', encoding='utf-8') as f:
        f.write(app_code)
    
    print(f"\nStarting Luid UI...")
    print(f"UI will be available at http://localhost:8501")
    
    # Launch the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501"]
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd)
    
    # Open web browser
    webbrowser.open("http://localhost:8501")
    
    print("\nLuid UI is now running. Press Ctrl+C in the terminal to stop.")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("Stopping Luid UI...")
        process.terminate()
    finally:
        # Clean up
        if os.path.exists(app_path):
            try:
                os.remove(app_path)
                os.rmdir(temp_dir)
            except:
                pass

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='X-Core AI Luid Interpretability Demo')
    parser.add_argument('--ui', action='store_true', help='Open the Luid UI')
    parser.add_argument('--open-ui', action='store_true', help='Skip demo and directly open the UI')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main demo function"""
    print("=== X-Core AI Interpretability Demo ===")
    
    # Parse command line arguments
    args = parse_args()
    
    # Create dummy config
    config = create_dummy_config()
    if args.config:
        # Load config from file if provided
        try:
            config.update(ConfigLoader.load_config(args.config))
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
    else:
        print("Using default configuration")
    
    # If open-ui flag is set, skip the demo and directly open the UI
    if args.open_ui:
        print("Opening Luid UI directly...")
        open_luid_ui()
        return
    
    # Run the Luid demo
    luid, core, sample_input, feature_names = demo_luid(config)
    
    print("\n=== Demo Completed Successfully ===")
    
    # Open UI if requested
    if args.ui:
        open_luid_ui(luid, core.model, sample_input, feature_names)

if __name__ == "__main__":
    main() 