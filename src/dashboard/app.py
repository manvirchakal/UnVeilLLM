import streamlit as st
import torch
from pathlib import Path
import sys
import plotly.graph_objects as go
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.model.model_loader import LlamaModelLoader
from src.analysis.activation_extractor import ActivationExtractor
from src.visualization.attention_vis import AttentionVisualizer
from src.analysis.neuron_analyzer import NeuronAnalyzer

def init_session_state():
    """Initialize session state variables"""
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = LlamaModelLoader()
        st.session_state.model, st.session_state.tokenizer = st.session_state.model_loader.load_model()
        st.session_state.activation_extractor = ActivationExtractor(
            st.session_state.model,
            st.session_state.tokenizer
        )
        st.session_state.attention_vis = AttentionVisualizer(
            st.session_state.model,
            st.session_state.tokenizer
        )
        st.session_state.neuron_analyzer = NeuronAnalyzer(
            st.session_state.model,
            st.session_state.activation_extractor
        )
    
    # Initialize analysis state
    if 'input_text' not in st.session_state:
        st.session_state.input_text = "Hello, world!"
    if 'attention_maps' not in st.session_state:
        st.session_state.attention_maps = None
    if 'current_layer' not in st.session_state:
        st.session_state.current_layer = 0
    if 'current_head' not in st.session_state:
        st.session_state.current_head = 0

def get_head_interpretation(layer: int, head: int, pattern_stats: Dict[str, float]) -> str:
    """Interpret the role of an attention head based on its patterns"""
    entropy = pattern_stats['entropy']
    sparsity = pattern_stats['sparsity']
    
    # Early layers (0-4): Local processing
    if layer < 5:
        if sparsity > 0.8:
            return "Local Feature Detector: Focuses on specific token patterns"
        elif entropy > 2.0:
            return "Context Gatherer: Broadly collects local context"
        else:
            return "Syntax Processor: Handles basic linguistic patterns"
    
    # Middle layers (5-9): Intermediate processing
    elif layer < 10:
        if sparsity > 0.8:
            return "Pattern Matcher: Identifies specific linguistic or semantic patterns"
        elif entropy > 2.0:
            return "Information Integrator: Combines multiple contexts"
        else:
            return "Semantic Processor: Processes meaning relationships"
    
    # Late layers (10-14): High-level processing
    else:
        if sparsity > 0.8:
            return "Task-Specific Focus: Concentrates on relevant information for the task"
        elif entropy > 2.0:
            return "Global Context Mixer: Integrates information across the full sequence"
        else:
            return "Output Formatter: Prepares for final representation"

def main():
    st.title("UnVeilLLM - Neural Network Interpretability Tool")
    
    # Initialize session state
    init_session_state()
    
    # Input section
    st.header("Input Text")
    input_text = st.text_area("Enter text to analyze:", st.session_state.input_text)
    
    # Analysis button
    if st.button("Analyze") or st.session_state.attention_maps is not None:
        if input_text != st.session_state.input_text or st.session_state.attention_maps is None:
            st.session_state.input_text = input_text
            st.session_state.attention_maps = st.session_state.attention_vis.get_attention_maps(input_text)
        
        # Visualization section
        st.header("Attention Visualization")
        layer = st.slider("Select Layer", 0, len(st.session_state.attention_maps)-1, st.session_state.current_layer)
        head = st.slider("Select Attention Head", 0, st.session_state.attention_maps[0].size(2)-1, st.session_state.current_head)
        
        # Update current layer and head
        st.session_state.current_layer = layer
        st.session_state.current_head = head
        
        # Create and display attention heatmap
        tokens = st.session_state.tokenizer.tokenize(input_text)
        fig = st.session_state.attention_vis.create_attention_heatmap(
            st.session_state.attention_maps, layer, head, tokens
        )
        
        # Add attention pattern analysis
        pattern_stats = st.session_state.attention_vis.analyze_attention_pattern(
            st.session_state.attention_maps, layer, head
        )
        
        st.plotly_chart(fig)
        
        # Display head interpretation
        st.subheader("Attention Head Analysis")
        interpretation = get_head_interpretation(layer, head, pattern_stats)
        st.info(f"Layer {layer}, Head {head} - Likely Role: {interpretation}")
        
        # Display attention pattern statistics in columns
        st.subheader("Attention Pattern Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pattern Statistics**")
            st.text(f"Maximum Attention: {pattern_stats['max_attention']:.4f}")
            st.text(f"Mean Attention: {pattern_stats['mean_attention']:.4f}")
            st.markdown("""
            *Maximum Attention*: Strongest single connection between tokens
            *Mean Attention*: Average attention weight across all connections
            """)
        
        with col2:
            st.markdown("**Pattern Characteristics**")
            st.text(f"Attention Entropy: {pattern_stats['entropy']:.4f}")
            st.text(f"Sparsity Score: {pattern_stats['sparsity']:.4f}")
            st.markdown("""
            *Attention Entropy*: How distributed the attention is (higher = more even)
            *Sparsity Score*: Fraction of low attention weights (< 0.1)
            """)
        
        # Add layer group explanation
        st.subheader("Layer Group Functions")
        st.markdown("""
        **Early Layers (0-4)**
        - Process basic features and local patterns
        - Handle syntax and token-level relationships
        - Build foundational representations
        
        **Middle Layers (5-9)**
        - Integrate information across longer spans
        - Process semantic relationships
        - Develop intermediate representations
        
        **Late Layers (10-14)**
        - Handle task-specific processing
        - Integrate global context
        - Prepare final output representations
        """)
        
        # Add neuron analysis section
        st.header("Neuron Analysis")
        
        # Extract activations for current layer
        layer_name = f"model.layers.{layer}"
        activations = st.session_state.activation_extractor.extract_activations(
            input_text,
            [layer_name]
        )[layer_name]
        
        # Get top neurons
        top_values, top_indices = st.session_state.neuron_analyzer.get_top_neurons(activations)
        
        # Create bar chart for neuron activations
        fig_neurons = go.Figure(data=[
            go.Bar(
                x=[f"Neuron {idx.item()}" for idx in top_indices[0]],
                y=[val.item() for val in top_values[0]],
                text=[f"{val.item():.4f}" for val in top_values[0]],
                textposition='auto',
            )
        ])
        
        fig_neurons.update_layout(
            title="Top 10 Most Active Neurons",
            xaxis_title="Neuron Index",
            yaxis_title="Activation Value",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Display visualization and text
        st.plotly_chart(fig_neurons)
        
        # Display neuron interpretation
        st.markdown("""
        **Neuron Activity Interpretation:**
        - Higher activation values indicate neurons that respond strongly to the input
        - Different neurons may specialize in detecting specific patterns or features
        - Early layer neurons often detect basic features
        - Later layer neurons typically combine features into higher-level concepts
        """)
        
        # Add correlation analysis for top neurons
        st.subheader("Top Neuron Correlations")
        
        with st.spinner('Computing correlations between top neurons...'):
            correlations = st.session_state.neuron_analyzer.find_correlated_neurons(
                activations,
                top_indices,
                threshold=0.5
            )
        
        if correlations:
            # Create correlation network graph for top neurons
            fig_correlations = go.Figure()
            
            # Create node positions in a circle
            import math
            n_neurons = len(top_indices[0])
            node_positions = {
                idx.item(): (
                    math.cos(2*math.pi*i/n_neurons), 
                    math.sin(2*math.pi*i/n_neurons)
                )
                for i, idx in enumerate(top_indices[0])
            }
            
            # Add edges (correlations)
            for i, j, corr in correlations:
                x0, y0 = node_positions[i]
                x1, y1 = node_positions[j]
                
                # Calculate midpoint for label position
                xmid = (x0 + x1) / 2
                ymid = (y0 + y1) / 2
                
                # Color interpolation: blue (-1) -> white (0) -> red (1)
                if corr > 0:
                    # Positive correlations: white to blue
                    color = f'rgba(65, 105, 225, {abs(corr)})'  # Royal blue with correlation strength
                else:
                    # Negative correlations: white to red
                    color = f'rgba(220, 20, 60, {abs(corr)})'   # Crimson red with correlation strength
                
                # Fixed width for cleaner look
                width = 1.5
                
                # Add line
                fig_correlations.add_trace(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='text',
                    text=f'Correlation: {corr:.3f}',
                    showlegend=False
                ))
                
                # Add correlation value label
                fig_correlations.add_annotation(
                    x=xmid,
                    y=ymid,
                    text=f'{corr:.2f}',
                    showarrow=False,
                    font=dict(size=10, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=2
                )
            
            # Add nodes
            for neuron_idx, (x, y) in node_positions.items():
                fig_correlations.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(size=15),
                    text=f'N{neuron_idx}',
                    textposition='top center',
                    hoverinfo='text',
                    showlegend=False
                ))
            
            fig_correlations.update_layout(
                title='Correlations Between Top 10 Most Active Neurons',
                showlegend=False,
                hovermode='closest',
                width=600,
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_correlations)
            
            st.markdown("""
            **Correlation Network Legend:**
            - Blue lines: Positive correlations (neurons activate together)
            - Red lines: Negative correlations (neurons have opposite activation patterns)
            - Line opacity shows correlation strength
            - Only correlations above 0.5 threshold are shown
            """)
        else:
            st.info("No strong correlations found between top neurons.")

if __name__ == "__main__":
    main() 