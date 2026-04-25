<p align="center">
    <img src="https://img.icons8.com/external-tal-revivo-regular-tal-revivo/96/external-readme-is-a-easy-to-build-a-developer-hub-that-adapts-to-the-user-logo-regular-tal-revivo.png" align="center" width="15%">
</p>
<p align="center"><h1 align="center"><code>❯ neural_network</code></h1></p>
<p align="center">
	<em>A deep-dive into the mathematical foundations of AI: Neural Network from Scratch.</em>
</p>

<p align="center">Built with:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
</p>
<br>

## 📍 Overview
**neural_network** is a custom implementation of a Multi-Layer Perceptron (MLP) built from scratch. It demonstrates the core principles of Deep Learning by translating **FEN (Forsyth-Edwards Notation)** strings into 78-dimensional tensors to classify chess board outcomes. The project includes benchmarks and multiple pre-configured network architectures.

---

## 📁 Project Structure

```sh
└── neural_network/
    ├── my_mlp.py                     # Core MLP engine & Backpropagation
    ├── params.py                     # CLI Argument & Environment handler
    ├── my_torch_generator.py         # Data processing & FEN Encoding
    ├── my_torch_analyzer.py          # Training visualization & Metrics
    ├── Benchmarks.pdf                # Performance analysis & Results
    ├── config/
    │   ├── my_torch_network_basic.conf   # Simple architecture
    │   ├── my_torch_network_medium.conf  # Intermediate architecture
    │   └── my_torch_network_full.conf    # Advanced deep architecture
    └── models/
        ├── my_torch_network_basic.nn     # Trained basic model
        ├── my_torch_network_medium.nn    # Trained medium model
        └── my_torch_network_full.nn      # Trained full model
```

🤖 Usage & Testing
The system is designed to be highly modular via command-line arguments.

1. Training a specific architecture:

Bash

❯ python3 my_mlp.py --train --file file.txt --load my_torch_network_medium.conf --save_file my_torch_network_medium.nn
2. Running Predictions:

Bash

❯ python3 my_mlp.py --predict --file file.txt --load my_torch_network_full.nn
3. Safety Features:
The engine includes a KeyboardInterrupt handler that automatically serializes the current "brain" state to a .nn file if the process is stopped, ensuring no training progress is lost.

📌 Project Roadmap
[X] Manual Gradient Descent: From-scratch implementation of backprop.

[X] Multi-Architecture Support: Basic, Medium, and Full configurations.

[X] FEN Transformation: Custom encoding of chess board states.

[X] Performance Benchmarking: Detailed PDF analysis of model results.

[ ] Optimization: Implementation of Adam/RMSProp optimizers.

🎗 License
This project is protected under the MIT License.
