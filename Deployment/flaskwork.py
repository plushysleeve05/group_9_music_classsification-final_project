import datetime
from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Set the paths to the uploads and model directories
uploads_path = r"C:\Users\ayoba\OneDrive - Ashesi University\Year 2 sem 2 fall\AI\FINAL PROJECT\Deployment\uploads"
model_path = r"C:\Users\ayoba\OneDrive - Ashesi University\Year 2 sem 2 fall\AI\FINAL PROJECT\Deployment\lsmt_model.h5"

# Load the trained model
model = load_model(model_path)

# Define the mapping of model output labels to genres
genre_mappings = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

def reshape_mfcc_for_rnn(X):
    num_rows = 1
    num_cols = 130
    num_features = X.shape[1]
    return np.reshape(X, (num_rows, num_cols, num_features))

@app.route('/')
def index():
    return render_template('musicwebsite.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'})

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'})

    try:
        # Ensure the uploads directory exists
        if not os.path.exists(uploads_path):
            os.makedirs(uploads_path)

        # Save the uploaded audio file
        audio_path = os.path.join(uploads_path, audio_file.filename)
        audio_file.save(audio_path)

        # Process the uploaded audio file using librosa
        audio, sr = librosa.load(audio_path, sr=22050, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_processed = np.mean(mfcc.T, axis=0)

        if mfcc_processed.shape[0] < 130:
            # Pad the features to match the desired shape
            mfcc_pad_width = 130 - mfcc_processed.shape[0]
            mfcc_processed = np.pad(mfcc_processed, pad_width=((0, mfcc_pad_width), (0, 0)), mode='constant', constant_values=0)
        elif mfcc_processed.shape[0] > 130:
            # Trim the features to match the desired shape
            mfcc_processed = mfcc_processed[:130, :]

        # Reshape MFCC data for LSTM input
        mfccs_reshaped = mfcc_processed[np.newaxis, ...]

        print(f"Shape of mfcc_processed: {mfcc_processed.shape}")
        print(f"Shape of mfccs_reshaped: {mfccs_reshaped.shape}")

        # Make predictions using the loaded LSTM model
        predicted_label = predict_genre(mfccs_reshaped)

        return jsonify({'predicted_genre': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_genre(features):
    # Perform prediction using the loaded LSTM model
    prediction = model.predict(features[np.newaxis, ...])
    # Get the index of the highest probability prediction
    predicted_label_index = np.argmax(prediction)
    # Return the genre corresponding to the predicted label index
    return genre_mappings[predicted_label_index]

if __name__ == '__main__':
    app.run(debug=True)