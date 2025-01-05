import torch
from typing import Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer

class ActivationExtractor:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_hooks = {}
        self.stored_activations = {}

    def register_hook(self, layer_name: str):
        """
        Register a forward hook for the specified layer
        """
        def hook_fn(module, input, output):
            # Handle tuple outputs (common in transformer layers)
            if isinstance(output, tuple):
                # Usually the first element contains the main output
                self.stored_activations[layer_name] = output[0].detach()
            else:
                self.stored_activations[layer_name] = output.detach()

        # Get the layer using named modules
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.activation_hooks[layer_name] = module.register_forward_hook(hook_fn)
                break

    def extract_activations(self, input_text: str, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract activations for specified layers given input text
        """
        # Clear previous activations
        self.stored_activations = {}
        
        # Register hooks for requested layers
        for layer_name in layer_names:
            self.register_hook(layer_name)

        # Tokenize and run inference
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)

        # Remove hooks
        for hook in self.activation_hooks.values():
            hook.remove()
        self.activation_hooks = {}

        return self.stored_activations 