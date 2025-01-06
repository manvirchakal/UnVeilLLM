# UnVeilLLM - Neural Network Interpretability Dashboard

A Python-based tool for exploring and understanding the internal mechanisms of neural networks, with initial focus on the Llama 3 family of models.

## Features

### Model Analysis
- Interactive visualization of attention patterns and layer activations
- Layer-wise Relevance Propagation (LRP) analysis
- Integrated Gradients attribution
- Token-level importance scoring
- Layer progression visualization

### Dashboard Components
- Real-time text input analysis
- Attention pattern heatmaps
- Token attribution visualization
- Layer-wise progression maps
- Interactive model behavior probing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/manvirchakal/UnVeilLLM
```

2. Create and activate a virtual environment:

On Linux or MacOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install PyTorch:

Navigate to the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install the appropriate version for your system.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:

```bash
streamlit run src/dashboard/app.py
```

## Features in Detail

### Attention Visualization
- Interactive heatmaps showing attention patterns
- Token-to-token relationship analysis
- Multi-layer attention comparison

### Attribution Analysis
- Layer-wise Relevance Propagation (LRP)
- Integrated Gradients
- Token importance scoring
- Layer progression visualization

### Model Behavior Analysis
- Neuron activation patterns
- Attention head roles
- Token relationship mapping

## Hardware Requirements

- GPU: NVIDIA RTX 3060 Ti (8GB VRAM)
- CPU: Intel i5 13400 or equivalent
- RAM: 32GB DDR5

## Optimization Features

- 4-bit quantization for efficient model loading
- Mixed precision computation
- Optimized memory management
- Selective layer analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Streamlit framework
- PyTorch ecosystem
- TransformerLens project