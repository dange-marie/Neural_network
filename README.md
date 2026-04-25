# Chess-State MLP: Neural Network from Scratch

A deep-dive into the mathematical foundations of Artificial Intelligence. This repository contains a complete implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch to analyze and classify chess board positions.

## 🧠 Technical Overview
Unlike projects that use high-level wrappers, this engine handles the heavy lifting of Deep Learning manually:
* **Manual Backpropagation:** Custom gradient calculation logic for weight and bias optimization.
* **Flexible Activation Suite:** Includes implementations for **Sigmoid, ReLU, Tanh, and Softmax** functions.
* **Dynamic Architecture:** Supports any number of hidden layers and neurons via JSON configuration.

## ♟️ Chess Integration
[cite_start]The model includes a specialized **FEN (Forsyth-Edwards Notation) Encoder** that converts chess board states into a 78-dimensional input vector:
* **Piece Mapping:** Assigns scores based on piece value and position.
* **Contextual Features:** Includes turn indicators (White/Black), castling rights, and en passant squares.
* **Classification:** Predicts 4 distinct outcomes: `Check`, `Nothing`, `Checkmate`, or `Stalemate`.

## 🛠️ Tech Stack
* **Language:** Python
* **Core Library:** NumPy (for optimized matrix operations)
* **Visualization:** Matplotlib (for real-time training cost tracking)

## 🚀 How It Works
1. **Configuration:** Define your network structure in a `.json` file (layers, learning rate, activations).
2. **Training:** The `MLP.train()` method processes large datasets using a generator to maintain memory efficiency.
3. **Prediction:** ```python
   # Example: Analyze a board state
   mlp = MLP("config.json")
   status = mlp.predict("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
   print(f"Board Status: {status}")
