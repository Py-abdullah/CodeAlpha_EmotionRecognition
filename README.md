# Emotion Recognition from Speech

This project is a speech emotion recognition system developed as part of the CodeAlpha Machine Learning Internship.

The model recognizes emotions from speech audio using **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and a **Convolutional Neural Network (CNN)** for classification.

## Project Objective

The objective of this project is to recognize different human emotions from speech audio using speech signal processing and deep learning techniques.

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The model is trained to classify the following emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Methodology

The project follows these main steps:

1. Load the RAVDESS speech dataset.
2. Identify emotion labels from the audio filenames.
3. Load and process the audio files.
4. Extract MFCC features from the speech audio.
5. Convert the audio features into a fixed-size format.
6. Encode emotion labels using Label Encoding.
7. Split the dataset into training and testing sets.
8. Normalize the MFCC features.
9. Build and train a CNN model.
10. Evaluate the model using test accuracy and classification metrics.
11. Generate a confusion matrix.
12. Test the trained model on new audio files.

## Feature Extraction

The main audio feature used in this project is **MFCC (Mel-Frequency Cepstral Coefficients)**.

MFCCs represent important characteristics of speech and are commonly used in speech and audio classification tasks.

The model uses:

- 40 MFCC coefficients
- Fixed sequence length of 130 time steps

## Model

A **1D Convolutional Neural Network (CNN)** is used for emotion classification.

The CNN contains:

- Conv1D layers
- Batch Normalization
- Max Pooling
- Dropout
- Dense layers
- Softmax output layer

Early stopping is also used during training to help prevent unnecessary training after validation performance stops improving.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Files

```text
CodeAlpha_EmotionRecognition/
│
├── Emotion_Recognition.ipynb
├── emotion_recognition_cnn.keras
├── label_encoder.pkl
├── mfcc_mean.npy
├── mfcc_std.npy
├── requirements.txt
└── README.md
