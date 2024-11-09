# Pneumonia-Detection-using-Chest-X-ray
This repository contains a deep learning model trained to detect pneumonia from chest X-ray images. The project leverages convolutional neural networks (CNNs) to classify X-ray images as either normal or indicative of pneumonia. This model was trained using a labeled dataset of chest X-ray images and provides a PyQt5-based frontend for easy image uploading and prediction.

Features:
        1. Deep learning model for binary classification (Normal/Pneumonia).
        2. Frontend GUI built with PyQt5 for image upload and prediction display.
        3. Model trained on chest X-ray images with high accuracy.

Install Required Dependencies
        Python 3.9 (recommended for PyQt5 compatibility)
        Install other packages with:
        # run this code in cmd. pip install -r requirements.txt
        Ensure packages include:
                1. TensorFlow/Keras
                2. PyQt5
                3. OpenCV
                4. Numpy, Matplotlib, etc.
Dataset:
        We used a publicly available dataset, Chest X-Ray Images (Pneumonia) from Kaggle. Download the dataset and place it in the data directory within the project          folder.
        Link to download Dataset:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Model Training:
        Data Preprocessing: Images were resized and normalized for better model performance.
        Model Architecture: A convolutional neural network (CNN) was used to achieve high accuracy in classifying pneumonia.
        Training: The model was trained using a batch size of 10 and saved as my_model.keras.
        Run Training Script in cmd: python train_model.py
        This script will save the trained model to the project directory.

Frontend Interface:
        This project includes a PyQt5-based graphical interface where users can:
        Upload chest X-ray images.
        View predictions (normal or pneumonia) with visual feedback.
        To start the GUI, run given code in cmd :
        python gui.py

Usage:
        Launch the GUI
        run in cmd:python gui.py

Upload Image:
        Use the upload button to select an X-ray image from your files.

Predict:
        Click "Predict" to classify the image and view the result.

Contributing:
        Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.


        THANK YOU......
        
