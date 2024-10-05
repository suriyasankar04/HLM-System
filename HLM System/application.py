from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import librosa
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Load the LSTM model
model = pickle.load(open('LSTMModel.pkl', 'rb'))

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for index page
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

# Function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to handle the file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'wav_file' not in request.files:
        return "No file part", 400

    file = request.files['wav_file']
    
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract MFCC features using librosa
        try:
            mfcc_features = extract_mfcc(file_path)
            mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Ensure correct shape for prediction
            prediction = model.predict(mfcc_features)
            predicted_class = np.argmax(prediction, axis=1)  # Get the predicted class
            # print(predicted_class[0])
            if(predicted_class[0] == 0):
                return "Angry"
            elif(predicted_class[0] == 1):
                return "Disgust"
            elif(predicted_class[0] == 2):
                return "Fear"
            elif(predicted_class[0] == 3):
                return "Happy"
            elif(predicted_class[0] == 4):
                return "Neutral"
            elif(predicted_class[0] == 5):
                return "Pleasent Suprise"
            elif(predicted_class[0] == 6):
                return "Sad"
            else:   
                return "Cannot be determine"
            # return jsonify({"filename": filename, "prediction": predicted_class[0]}), 200
         
        except Exception as e:
            return str(e), 500
    else:
        return "Invalid file type", 400

# Function to extract MFCC from audio file
def extract_mfcc(file_path):
    """
    Extract MFCC features from an audio file using librosa.
    :param file_path: Path to the audio file.
    :return: MFCC features as a numpy array.
    """
    y, sr = librosa.load(file_path, duration=3, offset=0.5)  # Loading audio
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)  # Extracting MFCC
    return mfcc

if __name__ == '__main__':
    app.run()
