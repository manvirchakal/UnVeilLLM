import torch
import plotly.graph_objects as go
from typing import List, Optional, Dict
import numpy as np

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_attention_maps(self, text: str) -> torch.Tensor:
        """Extract attention maps from the model"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Get attention tensors (layer, batch, heads, seq_len, seq_len)
        attention_maps = outputs.attentions
        return attention_maps

    def create_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        layer: int,
        head: int,
        tokens: List[str],
        title: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive heatmap for attention weights"""
        # Extract attention weights for specific layer and head
        attn = attention_weights[layer][0][head].cpu().numpy()
        
        # Create heatmap with improved formatting
        fig = go.Figure(data=go.Heatmap(
            z=attn,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate=(
                'Source Token: %{y}<br>'
                'Target Token: %{x}<br>'
                'Attention Weight: %{z:.4f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Update layout with more information and proper margins
        fig.update_layout(
            title={
                'text': title or f'Attention Pattern (Layer {layer}, Head {head})',
                'xanchor': 'center',
                'yanchor': 'top',
                'y': 0.95
            },
            xaxis_title="Target Tokens (being attended to)",
            yaxis_title="Source Tokens (doing the attending)",
            width=800,
            height=800,
            margin=dict(
                l=100,  # left margin
                r=100,  # right margin
                t=100,  # top margin
                b=150   # bottom margin
            ),
            annotations=[{
                'text': 'Darker colors indicate stronger attention weights',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': -0.2,
                'showarrow': False,
                'align': 'center',
                'font': {'size': 12}
            }]
        )
        
        return fig

    def analyze_attention_pattern(
        self,
        attention_weights: torch.Tensor,
        layer: int,
        head: int
    ) -> Dict[str, float]:
        """Analyze attention pattern characteristics"""
        attn = attention_weights[layer][0][head].cpu().numpy()
        
        # Normalize attention weights and handle numerical stability
        eps = 1e-10
        normalized_attn = attn.copy()
        normalized_attn = np.maximum(normalized_attn, eps)  # Ensure no zeros
        normalized_attn = normalized_attn / normalized_attn.sum()  # Normalize
        
        # Compute entropy only for non-zero values
        valid_mask = normalized_attn > eps
        entropy = 0.0
        if valid_mask.any():
            valid_attn = normalized_attn[valid_mask]
            entropy = -np.sum(valid_attn * np.log(valid_attn))
        
        return {
            "max_attention": float(np.max(attn)),
            "mean_attention": float(np.mean(attn)),
            "entropy": float(entropy),
            "sparsity": float(np.mean(attn < 0.1))
        } 