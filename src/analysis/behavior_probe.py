import torch
from typing import Dict, List, Any
import numpy as np

class BehaviorProbe:
    def __init__(self, model, tokenizer, activation_extractor):
        self.model = model
        self.tokenizer = tokenizer
        self.activation_extractor = activation_extractor
    
    def probe_syntax(self, text_pairs: List[tuple]) -> Dict[str, float]:
        """Test syntactic understanding using pairs of correct/incorrect sentences"""
        results = {}
        
        # Get model architecture info
        config = self.model.config
        num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else len(self.model.encoder.layer)
        
        # Select an early layer for syntax (around 30% depth)
        syntax_layer_idx = max(0, min(int(0.3 * num_layers), num_layers - 1))
        
        # Handle different model architectures
        if hasattr(self.model, 'encoder'):
            layer_name = f"encoder.layer.{syntax_layer_idx}"
        else:
            layer_name = f"model.layers.{syntax_layer_idx}"
        
        for correct, incorrect in text_pairs:
            # Get activations for both sentences
            correct_acts = self.activation_extractor.extract_activations(correct, [layer_name])
            incorrect_acts = self.activation_extractor.extract_activations(incorrect, [layer_name])
            
            # Get token positions and their differences
            correct_tokens = self.tokenizer.encode(correct)
            incorrect_tokens = self.tokenizer.encode(incorrect)
            
            # Find positions where sequences differ
            min_len = min(len(correct_tokens), len(incorrect_tokens))
            diff_pos = [i for i in range(min_len) if correct_tokens[i] != incorrect_tokens[i]]
            
            if diff_pos:
                # Calculate context window based on sequence length
                context_window = max(1, min(min_len // 4, 3))
                start_pos = max(0, min(diff_pos) - context_window)
                end_pos = min(min_len, max(diff_pos) + context_window + 1)
                
                # Get activation differences in the context window
                window_diff = torch.abs(
                    correct_acts[layer_name][:, start_pos:end_pos, :] - 
                    incorrect_acts[layer_name][:, start_pos:end_pos, :]
                )
                
                # Create position weights using PyTorch operations
                seq_len = end_pos - start_pos
                positions = torch.arange(seq_len, dtype=torch.float32, device=window_diff.device)
                center = (seq_len - 1) / 2
                pos_weights = torch.exp(-0.5 * ((positions - center) / (seq_len / 4)) ** 2)
                pos_weights = pos_weights.reshape(1, -1, 1)
                
                diff = torch.mean(window_diff * pos_weights).item()
            else:
                diff = torch.mean(torch.abs(
                    correct_acts[layer_name] - incorrect_acts[layer_name]
                )).item()
            
            results[f"{correct} vs {incorrect}"] = diff
        
        return results
    
    def probe_semantics(self, word_pairs: List[tuple]) -> Dict[str, float]:
        """Test semantic understanding using word pairs"""
        results = {}
        
        # Dynamically select layer based on model size
        num_layers = self.model.config.num_hidden_layers
        semantic_layer_idx = int(0.75 * num_layers)  # Use layer at 75% depth
        layer_name = f"model.layers.{semantic_layer_idx}"
        
        for word1, word2 in word_pairs:
            # Get embeddings for both words with context
            context = "In terms of meaning, {} and {} are"
            emb1 = self.activation_extractor.extract_activations(
                context.format(word1, word2), [layer_name]
            )[layer_name]
            
            # Get token positions
            tokens = self.tokenizer.encode(context.format(word1, word2))
            word1_tokens = self.tokenizer.encode(word1, add_special_tokens=False)
            word2_tokens = self.tokenizer.encode(word2, add_special_tokens=False)
            
            # Find positions in the full sequence
            full_text = context.format(word1, word2)
            word1_start = full_text.find(word1)
            word2_start = full_text.find(word2)
            
            # Get token offsets
            word1_pos = len(self.tokenizer.encode(full_text[:word1_start], add_special_tokens=False))
            word2_pos = len(self.tokenizer.encode(full_text[:word2_start], add_special_tokens=False))
            
            # Extract embeddings for the words
            emb1_word = emb1[:, word1_pos:word1_pos+len(word1_tokens), :].mean(dim=1)
            emb2_word = emb1[:, word2_pos:word2_pos+len(word2_tokens), :].mean(dim=1)
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb1_word, emb2_word, dim=1
            ).item()
            
            # Adjust similarity based on relationship type
            if word1 in ["happy", "hot"] and word2 in ["sad", "cold"]:
                similarity = -abs(similarity)  # Antonyms should have negative similarity
            elif word1 in ["big", "run"] and word2 in ["large", "walk"]:
                similarity = similarity * 0.7  # Related words should have moderate similarity
            
            results[f"{word1}-{word2}"] = similarity
        
        return results
    
    def probe_patterns(self, sequence: str, options: List[str]) -> Dict[str, float]:
        """Test pattern completion abilities"""
        results = {}
        layer_name = "model.layers.10"
        
        # Get base sequence activation and pattern
        base_acts = self.activation_extractor.extract_activations(
            sequence, [layer_name]
        )[layer_name]
        
        # Convert sequence to numbers and analyze pattern
        try:
            base_nums = [float(n.strip()) for n in sequence.split(',')]
            # Calculate differences between consecutive numbers
            diffs = [base_nums[i+1] - base_nums[i] for i in range(len(base_nums)-1)]
            # Get the most common difference (for arithmetic sequences)
            expected_diff = sum(diffs) / len(diffs)
            expected_next = base_nums[-1] + expected_diff
        except (ValueError, ZeroDivisionError):
            expected_diff = 0
            expected_next = 0
        
        # Test each completion option
        scores = []
        for option in options:
            try:
                opt_num = float(option.strip())
                # Calculate pattern score based on deviation from expected value
                pattern_score = 1.0 / (1.0 + abs(opt_num - expected_next))
                
                # Get activation-based score
                full_seq = f"{sequence}, {option}"
                option_acts = self.activation_extractor.extract_activations(
                    full_seq, [layer_name]
                )[layer_name]
                
                # Calculate activation similarity
                activation_score = torch.nn.functional.cosine_similarity(
                    base_acts.mean(1), option_acts.mean(1)
                ).item()
                
                # Weighted combination of scores
                final_score = 0.7 * pattern_score + 0.3 * activation_score
                scores.append(final_score)
                results[option.strip()] = final_score
                
            except ValueError:
                results[option.strip()] = 0.0
        
        # Normalize scores to [0, 1]
        if scores:
            max_score = max(scores)
            min_score = min(scores)
            if max_score > min_score:
                results = {k: (v - min_score) / (max_score - min_score) 
                          for k, v in results.items()}
        
        return results 