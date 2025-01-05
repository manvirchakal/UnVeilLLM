import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from typing import Tuple

class LlamaModelLoader:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self, load_4bit: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the Llama model with optional 4-bit quantization
        """
        # Initialize quantization config if needed
        quantization_config = None
        if load_4bit:
            quantization_config = bnb.nn.modules.Linear4bit.Config(
                compute_dtype=torch.float16,
                compress_statistics=True,
                quant_type="nf4"
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
            trust_remote_code=True
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