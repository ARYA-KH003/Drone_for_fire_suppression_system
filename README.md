Wildfire Detection – Supervised and Unsupervised Comparison (ResNet-50)

This project evaluates two wildfire image detection approaches using the same ResNet-50 backbone:

Supervised Model
ResNet-50 with a small classification head trained on labeled Fire / NoFire images.

Unsupervised Model
ResNet-50 feature embeddings followed by PCA, K-Means clustering, and threshold-based classification.

The objective is to examine whether an unsupervised pipeline can approximate the performance of the supervised classifier, especially in scenarios where annotated data is limited or expensive to obtain.

dataset: https://drive.google.com/file/d/1oHydL1ywUFzuUXAl-lmqD2ENF_iDgUyf/view?usp=sharing
