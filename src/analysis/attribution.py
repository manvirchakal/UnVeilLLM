from typing import Dict, List
import torch

class AttributionAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def integrated_gradients(self, 
        input_text: str,
        target_layer: str,
        steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients attribution"""
        # Tokenize input
        tokens = self.tokenizer(input_text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(input_ids).detach()
        baseline_embeddings = torch.zeros_like(embeddings)
        
        # Move to device
        embeddings = embeddings.to(self.model.device)
        baseline_embeddings = baseline_embeddings.to(self.model.device)
        
        # Accumulate gradients
        accumulated_grads = torch.zeros_like(embeddings)
        
        for alpha in torch.linspace(0, 1, steps):
            # Interpolate between baseline and input
            interpolated = (baseline_embeddings + alpha * (embeddings - baseline_embeddings)).clone()
            interpolated.requires_grad_(True)
            
            # Forward pass with interpolated embeddings
            outputs = self.model(inputs_embeds=interpolated)
            
            # Use prediction logits
            prediction = outputs.logits[:, -1, :].softmax(dim=-1)
            top_pred_prob = prediction.max()
            
            # Get gradients
            top_pred_prob.backward(retain_graph=True)
            accumulated_grads += interpolated.grad.detach()
            interpolated.grad = None
            
        # Calculate attribution scores
        attributions = (embeddings - baseline_embeddings) * accumulated_grads / steps
        
        # Get per-token attributions and normalize
        token_attributions = attributions.sum(dim=-1).squeeze(0)
        max_abs = torch.max(torch.abs(token_attributions))
        token_attributions = token_attributions / max_abs  # Scale to [-1, 1] range
        
        tokens_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            "tokens": tokens_list,
            "scores": token_attributions,
            "aggregated": token_attributions.abs().mean().item()
        }

    def layer_relevance_propagation(self,
        input_text: str,
        target_layer: str
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attribution scores"""
        # Forward pass
        tokens = self.tokenizer(input_text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.model.device)
        
        # Get all layer activations
        activations = {}
        def hook_fn(module, input, output, name):
            # Handle tuple outputs (common in attention layers)
            if isinstance(output, tuple):
                activations[name] = output[0].detach().to(torch.float32)  # Convert to float32
            else:
                activations[name] = output.detach().to(torch.float32)
            
        # Register hooks for all layers
        hooks = []
        for name, module in self.model.named_modules():
            if name.startswith("model.layers"):
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                ))
                
        # Forward pass to collect activations
        outputs = self.model(input_ids)
        logits = outputs.logits[:, -1, :].to(torch.float32)  # Convert to float32
        
        # Initial relevance is the predicted class score
        relevance = logits.max(dim=-1, keepdim=True)[0]
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Get target layer activation
        target_activation = activations[target_layer]
        
        # Project relevance to match target activation dimensions
        if relevance.size(-1) != target_activation.size(-1):
            proj_matrix = torch.ones(
                relevance.size(-1),
                target_activation.size(-1),
                device=relevance.device,
                dtype=torch.float32
            ) / target_activation.size(-1)
            relevance = torch.matmul(relevance, proj_matrix)
        
        # Compute token-level scores
        token_relevance = (target_activation * relevance).sum(dim=-1).squeeze()
        
        # Normalize scores
        max_abs = torch.max(torch.abs(token_relevance))
        if max_abs > 0:
            token_relevance = token_relevance / max_abs
        
        return {
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            "scores": token_relevance,
            "aggregated": token_relevance.abs().mean().item()
        }
        
    def _backward_lrp(self, relevance, activation, epsilon=1e-6):
        """Helper function for LRP backward pass"""
        # Convert both tensors to float32 for consistent dtype
        activation = activation.to(torch.float32)
        relevance = relevance.to(torch.float32)
        
        # Project relevance to match activation dimensions if needed
        if relevance.size(-1) != activation.size(-1):
            # Linear projection to match dimensions
            proj_matrix = torch.ones(
                relevance.size(-1), 
                activation.size(-1), 
                device=relevance.device,
                dtype=torch.float32
            ) / activation.size(-1)
            
            relevance = torch.matmul(relevance, proj_matrix)
        
        # Compute denominator with proper dimensions
        denominator = activation.sum(dim=-1, keepdim=True) + epsilon
        
        return (activation * relevance) / denominator 