import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple

class LlamaModelLoader:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def update_model(self, new_model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Update model with new model name and handle VRAM cleanup"""
        # Force CUDA synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Delete old model and force garbage collection
        if self.model is not None:
            self.model.cpu()  # Move model to CPU first
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Update model name and load new model
        self.model_name = new_model_name
        return self.load_model()

    def load_model(self, load_4bit: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the Llama model with optional 4-bit quantization
        """
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize quantization config if needed
        quantization_config = None
        if load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config if load_4bit else None,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        return self.model, self.tokenizer

    def get_model_info(self) -> dict:
        """
        Return basic model information
        """
        if self.model is None:
            raise ValueError("Model not loaded yet!")
        
        return {
            "model_name": self.model_name,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "memory_usage": f"{torch.cuda.memory_allocated() / 1024**2:.2f}MB"
        } 

    @staticmethod
    def get_available_models():
        return [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
    
    def update_model(self, new_model_name: str):
        """Update model with new model name"""
        self.model_name = new_model_name
        return self.load_model() 