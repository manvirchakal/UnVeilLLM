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
from src.analysis.behavior_probe import BehaviorProbe

def init_session_state():
    """Initialize session state variables"""
    # Initialize model-related states
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = LlamaModelLoader()
        st.session_state.model, st.session_state.tokenizer = st.session_state.model_loader.load_model()
        
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Initialize analysis states
    if 'input_text' not in st.session_state:
        st.session_state.input_text = "Hello, World!"
    
    if 'attention_maps' not in st.session_state:
        st.session_state.attention_maps = None
    
    if 'current_layer' not in st.session_state:
        st.session_state.current_layer = 0
    
    if 'current_head' not in st.session_state:
        st.session_state.current_head = 0
        
    # Initialize other components only after model is loaded
    if 'activation_extractor' not in st.session_state:
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
        st.session_state.behavior_probe = BehaviorProbe(
            st.session_state.model,
            st.session_state.tokenizer,
            st.session_state.activation_extractor
        )

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
    
    # Sidebar for navigation and model selection
    st.sidebar.title("Settings")
    
    # Model selection dropdown
    selected_model = st.sidebar.selectbox(
        "Select Model",
        LlamaModelLoader.get_available_models(),
        index=LlamaModelLoader.get_available_models().index(st.session_state.current_model)
    )
    
    # Update model if changed
    if selected_model != st.session_state.current_model:
        # Force CUDA synchronization first
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Clear old components and force them to CPU
        components_to_clear = [
            'model', 'tokenizer', 'activation_extractor', 
            'attention_vis', 'neuron_analyzer', 'behavior_probe'
        ]
        
        for component in components_to_clear:
            if component in st.session_state:
                if hasattr(st.session_state[component], 'cpu'):
                    st.session_state[component].cpu()
                del st.session_state[component]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Update model
        st.session_state.current_model = selected_model
        st.session_state.model, st.session_state.tokenizer = st.session_state.model_loader.update_model(selected_model)
        
        # Initialize new components
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
        st.session_state.behavior_probe = BehaviorProbe(
            st.session_state.model,
            st.session_state.tokenizer,
            st.session_state.activation_extractor
        )
        st.session_state.attention_maps = None
        
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Neuron & Attention Analysis", "Behavioral Analysis"]
    )
    
    if analysis_type == "Neuron & Attention Analysis":
        # Original analysis section
        st.header("Input Text")
        input_text = st.text_area("Enter text to analyze:", value="Hello, World!")
        
        # Rest of the original analysis code
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
                        text=f'Neuron {neuron_idx}',
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

    else:  # Behavioral Analysis
        # Add tabs for different probing types
        probe_tab1, probe_tab2, probe_tab3 = st.tabs([
            "Syntactic Analysis", 
            "Semantic Relations", 
            "Pattern Recognition"
        ])

        with probe_tab1:
            st.subheader("Syntax Understanding")
            st.markdown("""
            This section analyzes how the model detects grammatical errors. 
            
            **How to use:**
            1. The chart shows example pairs of correct/incorrect sentences
            2. Higher values mean the model strongly detects the grammar mistake
            3. Try your own sentences in the comparison tool below
            4. The correct sentence should go in the left box
            5. The grammatically incorrect version in the right box
            """)
            
            # Example sentence pairs for testing
            syntax_pairs = [
                ("The cat is sleeping", "The cat are sleeping"),
                ("She runs fast", "She run fast"),
                ("I saw the red car", "I saw the car red")
            ]
            
            # Run syntax probe and visualize results
            syntax_results = st.session_state.behavior_probe.probe_syntax(syntax_pairs)
            
            # Create bar chart for syntax differences
            fig_syntax = go.Figure(data=[
                go.Bar(
                    x=list(syntax_results.keys()),
                    y=list(syntax_results.values()),
                    text=[f"{val:.3f}" for val in syntax_results.values()],
                    textposition='auto',
                )
            ])
            
            fig_syntax.update_layout(
                title="Activation Differences for Syntactic Variations",
                xaxis_title="Sentence Pairs",
                yaxis_title="Activation Difference",
                height=400,
                margin=dict(l=50, r=50, t=50, b=100)
            )
            
            st.plotly_chart(fig_syntax)
            
            st.markdown("""
            **Interpretation:**
            - Higher values indicate stronger model reaction to grammatical errors
            - Lower values suggest the model is less sensitive to the grammatical mistake
            - Note: This measures syntax (grammar) differences, not meaning differences
            """)

            # Add interactive syntax comparison
            st.subheader("Compare Custom Sentences")
            col1, col2 = st.columns(2)
            
            with col1:
                sentence1 = st.text_input(
                    "Enter first sentence:",
                    value="The cat is sleeping"
                )
            
            with col2:
                sentence2 = st.text_input(
                    "Enter second sentence:",
                    value="The cat are sleeping"
                )
            
            if st.button("Compare Syntax"):
                # Include baseline pairs for normalization context
                all_pairs = [
                    ("The cat is sleeping", "The cat are sleeping"),
                    ("She runs fast", "She run fast"),
                    ("I saw the red car", "I saw the car red"),
                    (sentence1, sentence2)  # Add custom pair
                ]
                
                # Run syntax probe on all pairs
                all_results = st.session_state.behavior_probe.probe_syntax(all_pairs)
                
                # Get the normalized score for custom pair
                similarity = all_results[f"{sentence1} vs {sentence2}"]
                
                # Create gauge chart for syntax difference
                fig_syntax_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = similarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 0.5]},  # Adjust range based on typical activation differences
                        'bar': {'color': "lightblue"},
                        'steps': [
                            {'range': [0, 0.1], 'color': "lightgreen"},
                            {'range': [0.1, 0.3], 'color': "yellow"},
                            {'range': [0.3, 0.5], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': similarity
                        }
                    }
                ))
                
                fig_syntax_gauge.update_layout(
                    title={
                        'text': "Syntactic Difference Score",
                        'y': 0.8,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=300,
                    margin=dict(l=50, r=50, t=100, b=50)
                )
                
                st.plotly_chart(fig_syntax_gauge)

        with probe_tab2:
            st.subheader("Semantic Understanding")
            st.markdown("""
            This section measures how the model understands word meanings and relationships.
            
            **How to use:**
            1. The heatmap shows similarity between word pairs
            2. Red indicates opposite meanings (-1)
            3. Blue indicates similar meanings (+1)
            4. Gray indicates unrelated words (0)
            5. Try your own word pairs in the comparison tool below
            """)
            
            # Word pairs for semantic testing
            word_pairs = [
                ("happy", "sad"),
                ("hot", "cold"),
                ("big", "large"),
                ("run", "walk")
            ]
            
            # Run semantic probe
            semantic_results = st.session_state.behavior_probe.probe_semantics(word_pairs)
            
            # Create heatmap for semantic relationships
            fig_semantic = go.Figure(data=go.Heatmap(
                z=[[val] for val in semantic_results.values()],
                y=list(semantic_results.keys()),
                colorscale='RdBu',
                text=[[f"{val:.3f}"] for val in semantic_results.values()],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorbar_title="Similarity",
                zmin=-1,  # Set minimum value for color scale
                zmax=1    # Set maximum value for color scale
            ))
            
            fig_semantic.update_layout(
                title="Word Pair Semantic Similarities",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_semantic)
            
            st.markdown("""
            **Interpretation:**
            - Values close to 1 indicate synonyms
            - Values close to -1 indicate antonyms
            - Values near 0 indicate unrelated words
            """)

            # Add after the existing semantic heatmap visualization
            st.subheader("Interactive Semantic Comparison")

            col1, col2 = st.columns(2)
            with col1:
                word1 = st.text_input("Enter first word:", value="happy")
            with col2:
                word2 = st.text_input("Enter second word:", value="sad")

            if st.button("Compare Words"):
                # Compute similarity using the existing probe
                similarity = st.session_state.behavior_probe.probe_semantics([(word1, word2)])[f"{word1}-{word2}"]
                
                # Create gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = similarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "red"},
                            {'range': [-0.5, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "blue"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': similarity
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    title={
                        'text': f"Semantic Similarity between '{word1}' and '{word2}'",
                        'y':0.8,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=300,
                    margin=dict(l=50, r=50, t=100, b=50)
                )
                
                st.plotly_chart(fig_gauge)

        with probe_tab3:
            st.subheader("Pattern Recognition")
            st.markdown("""
            This section tests the model's ability to recognize and continue numerical patterns.
            
            **How to use:**
            1. Enter a sequence of numbers separated by commas
            2. Enter possible continuations separated by commas
            3. The model will score each continuation
            4. Lower scores indicate better pattern matches
            
            **Examples:**
            - Arithmetic: "2, 4, 6, 8" → "10"
            - Geometric: "2, 4, 8, 16" → "32"
            - Fibonacci: "1, 1, 2, 3, 5" → "8"
            """)
            
            # Input for sequence
            sequence = st.text_input(
                "Enter a sequence pattern:",
                value="2, 4, 6, 8"
            )
            
            # Input for possible completions
            options = st.text_input(
                "Enter possible completions (comma-separated):",
                value="10, 12, 9"
            ).split(",")
            options = [opt.strip() for opt in options]
            
            if st.button("Analyze Pattern"):
                pattern_results = st.session_state.behavior_probe.probe_patterns(sequence, options)
                
                # Create bar chart for pattern completion scores
                fig_pattern = go.Figure(data=[
                    go.Bar(
                        x=list(pattern_results.keys()),
                        y=list(pattern_results.values()),
                        text=[f"{val:.3f}" for val in pattern_results.values()],
                        textposition='auto',
                    )
                ])
                
                fig_pattern.update_layout(
                    title="Pattern Completion Analysis",
                    xaxis_title="Possible Completions",
                    yaxis_title="Continuation Score",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig_pattern)
                
                st.markdown("""
                **Interpretation:**
                - Lower scores suggest better pattern continuation
                - Higher scores indicate more deviation from the pattern
                - Compare scores to understand model's pattern preference
                """)

if __name__ == "__main__":
    main() 