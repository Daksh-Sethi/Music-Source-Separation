# 🎵 Music Source Separation Using U-Net

This repository implements a **Music Source Separation** project using a U-Net-based deep learning model. The goal is to separate **drum tracks** from a mixture of audio tracks in songs using the **MUSDB dataset**. This project includes both the training code and the testing/demo code to apply the trained model on any song.

---

## **Table of Contents**
- [Overview](#overview)
- [Files in the Repository](#files-in-the-repository)
- [Methodology](#methodology)
  - [Dataset](#dataset)
  - [Data Preparation](#data-preparation)
  - [Network Architecture](#network-architecture)
  - [Training](#training)
  - [Reconstruction](#reconstruction)
  - [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

---

## **Overview**
Music source separation is the task of isolating different tracks, such as vocals, drums, and bass, from a mixture audio file. With recent advancements in neural networks and data availability in the audio domain, models like **WaveNet** and **U-Net** have shown great promise in achieving this. This project uses a U-Net-like structure to separate **drums** from audio tracks in the **MUSDB dataset**.

This system can benefit:
- **Music producers**: Isolate individual components for remixing or audio editing.
- **Researchers and developers**: Study and improve source separation techniques.
- **Music enthusiasts**: Extract and analyze isolated tracks from their favorite songs.

---

## **Files in the Repository**
1. **`music_source_separation.ipynb`**:
   - Contains the code for training the U-Net model on the MUSDB dataset.
   - Includes preprocessing, training, and evaluation.

2. **`project_elc.ipynb`**:
   - A demo notebook for testing the trained model on custom songs.
   - Takes a song as input, processes it, and outputs the isolated drum track.
---

## **Methodology**

### **Dataset**
We use the **MUSDB** dataset, a comprehensive source separation dataset containing 150 professionally produced songs across various genres like pop, rock, and hip-hop. Each song has:
- **Five tracks**: Mix, drums, bass, vocals, and others.
- **Split**: 100 songs for training and 50 for testing.

### **Data Preparation**
1. Extracted mixture and drum files for every song.
2. Processed audio using the `librosa` library:
   - Resampled to 22,050 Hz.
   - Split into 1-second blocks with 0.5-second overlap.
   - Transformed to **STFT (Short-Time Fourier Transform)** matrices with dimensions `(251, 177, 2)` (real and imaginary parts).

### **Network Architecture**
Our U-Net-inspired architecture:
- **Input shape**: `(251, 177, 2)`.
- **9 convolutional layers**:
  - Skip connections to preserve spatial information.
  - 32 to 512 filters in the encoder-decoder stages.
- **Output shape**: Same as input, preserving real and imaginary parts.

### **Training**
- Batch size: 3 songs per batch.
- Optimizer: ADAM with a learning rate of 0.001.
- Loss function: Mean Squared Error.
- Trained for **20 epochs** per batch using Kaggle T4 GPU.
- Validation split: 20%.

### **Reconstruction**
1. Combined real and imaginary parts to reconstruct the STFT matrix.
2. Applied inverse STFT to convert back to the time domain.
3. Saved the reconstructed audio as an MP3 file for listening.

### **Evaluation**
- **Metric**: Mean Squared Error (MSE) during training.

---

## **How to Run**

### ** You can open kaggle and load the pynb files and run the code**

