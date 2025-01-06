# Semantic Understanding Findings

## Unexpected Word Pair Similarities

### 1. Emotions and Opposites
- **Anger-Joy**: 0.678 similarity despite being opposing emotions
- **Love-Hate**: 0.688 similarity despite being opposing concepts
- **Happy-Sad**: -0.765 (expected negative correlation)

#### Analysis
The model shows strong positive correlations between seemingly opposite emotional concepts (anger-joy, love-hate) while maintaining expected negative correlation for basic opposites (happy-sad). This suggests:
- Emotional concepts are grouped by intensity rather than valence
- Complex emotions share more contextual patterns than simple opposites
- The model captures linguistic usage patterns over human intuitive relationships

### 2. Physical Concepts
- **Light-Dark**: 0.727 similarity despite being opposites
- **Fast-Slow**: 0.720 similarity despite being opposites
- **Hot-Cold**: -0.688 (expected negative correlation)

#### Analysis
Physical concepts show varying patterns:
- Basic temperature opposites (hot-cold) maintain negative correlation
- Motion and illumination pairs show unexpected positive correlation
- Suggests these terms often appear in similar contexts despite opposite meanings

### Technical Implementation Reference
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
This finding highlights how language models may group concepts based on structural/contextual similarities rather than human-intuitive semantic relationships. The model appears to weight the intensity/arousal dimension of emotions more 
heavily than their positive/negative valence.

### Key Insights
1. The model groups concepts based on:
   - Contextual co-occurrence
   - Usage patterns in language
   - Intensity of concepts
2. Simple opposites (hot-cold, happy-sad) maintain expected negative correlations
3. Complex concept pairs show unexpected positive correlations

### References
- Implementation: `src/analysis/behavior_probe.py` lines 70-117
- Semantic probing methodology: lines 105-107