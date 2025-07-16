🧠 Face Recognition System using OpenCV

A real-time face recognition system built with Python and OpenCV, utilizing the LBPH (Local Binary Patterns Histograms) algorithm for accurate and efficient face identification.

This system allows users to:

    Capture face images from a webcam

    Train a facial recognition model

    Test the model on new images

    Display prediction results with confidence levels

🖼️ Demo

<img width="630" height="565" alt="Screenshot 2025-07-16 133840" src="https://github.com/user-attachments/assets/4d05fb6f-c966-483e-9626-9718a32dfb7c" />


   
🔍 Features

    ✅ Real-time face detection using Haar Cascades

    ✅ Face recognition using OpenCV’s LBPH algorithm

    ✅ Model training from captured face data

    ✅ Prediction on test images with confidence scoring

    ✅ Lightweight and fast — works even without a GPU

🛠️ Technologies Used

    Python 3.x

    OpenCV

    LBPH Face Recognizer

    Haarcascade Classifier

📦 Requirements

Install required libraries using pip:

pip install opencv-python

📚 How it Works

    Face Detection
    Detects faces using Haar Cascades in frames captured from the webcam.

    Face Capture & Dataset Generation
    Saves multiple images of a user's face into a dataset directory for training.

    Model Training
    Trains the LBPHFaceRecognizer using the collected face dataset.

    Prediction
    Compares a test image to the trained model and returns the most probable match with a confidence score.

🚧 Future Improvements

    🔄 Add support for multiple users with dynamic name mapping

    📋 Integrate with a database for attendance logging

    🎨 Build a GUI using Tkinter or PyQt

    📈 Improve accuracy with face alignment and preprocessing

👤 Author

Omar Alshargawi

