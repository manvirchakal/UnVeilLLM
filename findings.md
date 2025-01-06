# Semantic Understanding Findings

## Unexpected Similarity: Anger and Joy

### Observation
The model shows unexpected similarity between seemingly opposite emotions "anger" and "joy". This suggests the model may be capturing shared characteristics rather than their opposing valence.

### Analysis
#### Shared Characteristics
- Both are high-intensity emotions
- Both appear in similar linguistic contexts
- Both represent strong emotional states

### Technical Evidence
#### Code Implementation

#### Layer Selection for semantic probe(75% depth)
```python
semantic_layer_idx = int(0.75 * num_layers)
layer_name = f"model.layers.{semantic_layer_idx}"
```

#### Context template
```python
context_template = "In terms of meaning, {} and {} are"
```

#### Similarity computation
```python
cossine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1).mean()
```

### Future Investigation
1. **Comparative Analysis**
   - Compare with other emotion pairs
   - Test intensity vs valence hypothesis
   
2. **Technical Deep Dive**
   - Analyze attention patterns
   - Examine layer-wise representations
   - Study context effects

### Key Insight
This finding highlights how language models may group concepts based on structural/contextual similarities rather than human-intuitive semantic relationships. The model appears to weight the intensity/arousal dimension of emotions more heavily than their positive/negative valence.

### References
- Implementation: `src/analysis/behavior_probe.py`
- Semantic probing: lines 70-115
- Layer selection: line 76
- Context template: line 81
- Similarity computation: lines 105-107
