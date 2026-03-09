# Defect-detection
Industrial Defect Detection using Machine Learning

Built a neural network pipeline from scratch using NumPy, including a modular Layer API, dense layers with manual backpropagation, ReLU activations, and gradient-descent parameter updates. Implemented a full training loop with mini-batch learning on a synthetic regression task and visualized both the learned function and the learning curve. Extended the work to computer-vision fundamentals by implementing zero-padding and multi-channel 2D convolution over image batches (stride support), mirroring the core operations behind CNNs.

This project demonstrates the ability to build and train neural networks from the ground up, focusing on the core mechanics that modern deep learning frameworks automate. Instead of relying on high-level APIs, I implemented the key components manually to deeply understand forward passes, backpropagation, optimization, and convolution operations (the building blocks behind CNN-based defect detection systems).

What I built - 

A minimal neural-network “layer” framework

Designed a clean Layer interface with forward, backward, and update to structure learning systems in a modular way.

A fully-connected neural network (MLP) from scratch (NumPy)

Implemented a custom Linear/Dense layer (forward pass, gradient computation, parameter updates).

Implemented ReLU activation with correct backward gradient masking.

Composed layers into a simple 2-layer neural network and verified shapes/gradients through sanity checks.

A complete training pipeline

Built a mini-batch gradient descent training loop.

Trained the model on a synthetic regression dataset and tracked training loss.

Visualized:

model predictions vs. training 

learning curves (loss over epochs)

Computer vision building blocks (NumPy)

Implemented zero-padding for batched multi-channel images.

Implemented multi-channel 2D convolution over a batch of images with configurable stride (core CNN operation).

This repo is a practical proof that I understand how deep learning works under the hood—from tensor shapes to gradient flow—while writing clean, testable numerical code.

Skills & tools demonstrated (Recruiter-facing)
Machine Learning / Deep Learning Fundamentals
Neural networks (MLP), forward propagation & backpropagation
Manual gradient derivation + implementation
Mini-batch gradient descent optimization
Weight initialization (He initialization)
Activation functions (ReLU)
Loss computation & training stability concepts
Computer Vision Foundations
Image padding (zero-padding)
Multi-channel convolution (batch-wise)
Stride handling and convolution output dimension reasoning
Programming / Engineering
Python
NumPy (vectorized numerical computing, tensor/array shape management)
Matplotlib (visualization of results + learning curves)
tqdm (training progress monitoring)
Modular design patterns (layer abstractions, composable components)
Debugging numerical code (sanity checks for shapes/gradients)
