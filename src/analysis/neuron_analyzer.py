import torch
from typing import List, Dict, Tuple
import numpy as np
from torch import nn

class NeuronAnalyzer:
    def __init__(self, model, activation_extractor):
        self.model = model
        self.activation_extractor = activation_extractor
        
    def get_top_neurons(
        self,
        activations: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the most active neurons"""
        # Calculate mean activation across sequence length
        mean_activations = torch.mean(activations, dim=1)
        
        # Get top-k neurons
        top_values, top_indices = torch.topk(mean_activations, k=top_k)
        return top_values, top_indices
    
    def compute_neuron_sensitivity(
        self,
        layer_name: str,
        input_text: str,
        num_samples: int = 100
    ) -> torch.Tensor:
        """Compute neuron sensitivity using gradient-based attribution"""
        # Get original activations
        activations = self.activation_extractor.extract_activations(
            input_text,
            [layer_name]
        )[layer_name]
        
        # Convert to float and enable gradients
        activations = activations.float().requires_grad_(True)
        
        # Initialize sensitivity scores
        sensitivity = torch.zeros_like(activations)
        
        # Compute gradients
        for i in range(num_samples):
            noise = torch.randn_like(activations) * 0.1
            noisy_activations = activations + noise
            
            # Compute output difference
            output_diff = torch.sum(torch.abs(noisy_activations - activations))
            
            # Compute gradients
            grad = torch.autograd.grad(output_diff, activations, retain_graph=True)[0]
            sensitivity += torch.abs(grad)
            
        return sensitivity / num_samples

    def find_correlated_neurons(
        self,
        activations: torch.Tensor,
        top_indices: torch.Tensor,  # Pass in indices of top neurons
        threshold: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """Find correlations between top neurons"""
        # Get dimensions and reshape activations
        batch_size, seq_len, num_neurons = activations.size()
        flat_activations = activations.reshape(-1, num_neurons)
        
        # Select only the top neurons
        top_neuron_activations = flat_activations[:, top_indices[0]]
        
        # Transpose to get neurons as rows for correlation computation
        top_neuron_activations = top_neuron_activations.T.float()
        
        # Compute correlation matrix for top neurons
        # Normalize the activations
        top_neuron_activations = top_neuron_activations - top_neuron_activations.mean(dim=1, keepdim=True)
        norms = torch.norm(top_neuron_activations, dim=1, keepdim=True)
        top_neuron_activations = top_neuron_activations / (norms + 1e-8)
        
        # Compute correlation matrix
        corr_matrix = torch.mm(top_neuron_activations, top_neuron_activations.T)
        
        # Find correlated pairs
        correlations = []
        for i in range(len(top_indices[0])):
            for j in range(i + 1, len(top_indices[0])):
                corr_value = corr_matrix[i, j].item()
                if abs(corr_value) > threshold:
                    correlations.append((
                        top_indices[0][i].item(),
                        top_indices[0][j].item(),
                        corr_value
                    ))
        
        return correlations 