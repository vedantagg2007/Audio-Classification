# Audio Classification
Classifying environmental sounds

# Environmental Audio Classification

A deep learning project for classifying environmental sounds from 10-second audio recordings using different signal representations and convolutional neural networks.

The project focuses on three classes: **jackhammer, siren, and chainsaw**. Audio is sampled at **16 kHz** and processed into multiple representations to compare how feature extraction affects classification performance. A top-k approach is applied to balance data, with k = 3.

I implemented and evaluated three CNN-based approaches:

- **Raw Waveform 1D CNN** – trains directly on time-domain audio samples
- **DCT 2D CNN** – uses Discrete Cosine Transform features
- **FFT/STFT 2D CNN** – uses frequency-domain spectrogram representations

The models were trained on a balanced dataset with **400 training, 75 validation, and 75 test samples per class**.

### Technologies

Python, PyTorch, NumPy, SciPy, Librosa, Scikit-learn, Matplotlib