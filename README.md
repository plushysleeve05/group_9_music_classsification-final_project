# Audio Genre Classification with LSTM Model Deployment

This repository contains code and instructions to deploy an LSTM-based machine learning model for audio genre classification using Flask.

## Overview

This project aims to deploy an LSTM model trained on audio data to predict the genre of uploaded audio files. The deployed model utilizes MFCC features extracted from audio files to make predictions.

## Setup

### Requirements

- Python (version 3.11.0)
- Libraries (list libraries and versions)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Ensure the pre-trained LSTM model file (`gru_model.h5`) is available.
2. Run the Flask application:

    ```bash
    python app.py
    ```

3. Access the application via a web browser at `http://localhost:5000`.

### File Structure

- `/models`: Contains the pre-trained LSTM model file.
- `/static`, `/templates`: Frontend files for the web application.
- `app.py`: Flask application for model deployment.
- `utils.py`: Utility functions for audio processing and prediction.

## How to Use

1. Visit the web application in your browser.
2. Upload an audio file (supported formats: mp3, wav, etc.).
3. Click the 'Predict' button to get the predicted genre for the uploaded audio file.

## Deployment

This project can be deployed using various platforms like Heroku, AWS, GCP, etc. Follow the platform-specific deployment instructions.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.



## Acknowledgments

- Mention any contributors or resources you used or were inspired by.
- Acknowledge libraries, frameworks, or tools utilized in the project.



For a visual walkthrough of how to deploy this model, check out this [YouTube video tutorial](https://youtu.be/CO6KYKTlhJ4).

