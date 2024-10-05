# HLM-System
A Hate Language Monitoring System using Machine Learning detects and flags offensive or harmful language in online platforms. It processes text through cleaning, tokenization, and feature extraction (e.g., word embeddings). Using models like RNN, the system classifies content, enabling real-time monitoring to reduce toxic communication.

# OVERVIEW
The Hate Speech Monitoring System project is designed to develop a comprehensive framework for detecting and mitigating hate speech in online communications, responding to the pressing need for effective solutions due to the rising incidence of hate speech on social media platforms. This project employs advanced machine learning techniques, particularly Long Short-Term Memory (LSTM) networks, in conjunction with Mel Frequency Cepstral Coefficients (MFCC), to enhance the accuracy of hate speech detection by focusing on audio signals. By capturing nuanced acoustic characteristics, the system aims to identify subtle variations that signify hate language, thereby improving detection capabilities. The project involves gathering a diverse dataset encompassing both hate speech and neutral speech to ensure representation across various contexts, alongside implementing preprocessing techniques to clean and normalize the data for effective model training. Furthermore, it utilizes MFCC for feature extraction to highlight significant temporal and spectral properties of the audio recordings, and it designs an LSTM-based architecture specifically tailored for hate speech detection, enabling the model to learn complex patterns and relationships within the data. The model's performance will be assessed using metrics such as accuracy, precision, recall, and F1-score, with a focus on optimizing it to perform well in scenarios characterized by imbalanced data. Finally, the project will develop a user-friendly interface to allow stakeholders to utilize the monitoring system effectively, enabling real-time operation for timely detection and feedback on hate speech. Through this initiative, the project aims not only to provide a robust solution for identifying hate speech but also to contribute to broader efforts aimed at fostering respectful online discourse and creating a safer digital environment for all users.

# DATASET
Toronto emotional speech set (TESS)

There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.

# MODEL PERFORMANCE

A model performance graph typically includes plots like the accuracy/loss curve, which shows how the model's accuracy and error evolve over training epochs for both training and validation sets, helping detect overfitting.

![image](https://github.com/user-attachments/assets/a7a72bef-38df-4d82-9352-76b44a7b2888)

The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format

# FEATURE EXTRACTION

MFCC (Mel-Frequency Cepstral Coefficients) using the librosa library is a technique for extracting features from audio that represent its timbral characteristics. It mimics how humans perceive sound by focusing on the most important frequencies.

Steps in MFCC Extraction using librosa:
Loading the Audio: The audio is loaded using librosa.load(), which returns the audio time series and the sampling rate.
MFCC Calculation: librosa.feature.mfcc() extracts MFCCs, where you can specify parameters like the number of MFCCs to compute.
Visualization: MFCCs can be plotted using librosa.display.specshow() to visualize how these coefficients change over time.
MFCCs are widely used in tasks like speech recognition and audio classification because they effectively capture the perceptual features of sound.

# MODEL

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to capture and remember long-term dependencies in sequential data. Unlike standard RNNs, LSTMs can retain information over long periods by using memory cells and gates (input, forget, and output gates) to control the flow of information. This makes LSTMs particularly useful for tasks like time series prediction, natural language processing, and speech recognition, where the model needs to remember past information to make accurate predictions

![image](https://github.com/user-attachments/assets/590ce756-c717-4199-89fb-7511568471e1)

# VISUALISATION
Visualizing the TESS dataset (Toronto Emotional Speech Set) involves representing the emotions conveyed in speech through bar chart
![image](https://github.com/user-attachments/assets/0524ca30-9743-4b71-8772-fc93a147ff3b)

# DEMO
A demo application is created using html and python
![image](https://github.com/user-attachments/assets/a41ad42e-c980-48db-bf02-eb83fe5b4c9e)
