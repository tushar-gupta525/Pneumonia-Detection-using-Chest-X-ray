# Pneumonia-Detection-using-Chest-X-ray
This repository contains a deep learning model trained to detect pneumonia from chest X-ray images. The project leverages convolutional neural networks (CNNs) to classify X-ray images as either normal or indicative of pneumonia. This model was trained using a labeled dataset of chest X-ray images and provides a PyQt5-based frontend for easy image uploading and prediction.

Features:
        1. Deep learning model for binary classification (Normal/Pneumonia).
        2. Frontend GUI built with PyQt5 for image upload and prediction display.
        3. Model trained on chest X-ray images with high accuracy.

Follow Step to create this project:        

Use Code1.ipynb file and use jupyter Notebook for this project. Follow link provided below to setup jupyuter Notebook.
Link-1:https://youtu.be/ClTWPoDHY_s?si=-HBEN7RNQoWfB2S3
Link-2:https://youtu.be/r8BXJdE9ChE?si=lMKIAmuXQSqTEkNt

Step-1: Install Required Dependencies:
        Python 3.9 (recommended for PyQt5 compatibility)
        Install other packages with:
        # run this code in cmd. pip install -r requirements.txt
        Ensure packages include:
                1. TensorFlow/Keras
                2. PyQt5
                3. OpenCV
                4. Numpy, Matplotlib, etc.
Step-2: Dataset:
        We used a publicly available dataset, Chest X-Ray Images (Pneumonia) from Kaggle. Download the dataset and place it in the data directory within the project          folder.
        Link to download Dataset:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Step-3: Model Training:
        Data Preprocessing: Images were resized and normalized for better model performance.
        Model Architecture: A convolutional neural network (CNN) was used to achieve high accuracy in classifying pneumonia.
        Training: The model was trained using a batch size of 10 and saved as my_model.keras.
        Run Training Script in cmd: python train_model.py
        This script will save the trained model to the project directory.

Now use Sublime Text for frontend interface. Follow link provided below to setup Sublime Text.
Link:https://youtu.be/yVK25kXNuzw?si=GjgSNqXAd2YcQFAD

Step-4: Frontend Interface:
        This project includes a PyQt5-based graphical interface where users can:
        Upload chest X-ray images.
        View predictions (normal or pneumonia) with visual feedback.
        To start the GUI, run given code in cmd :
        python gui.py

Step-4: Usage:
        Launch the GUI
        run in cmd:python gui.py

Step-5: Upload Image:
        Use the upload button to select an X-ray image from your files.

Step-6: Predict:
        Click "Predict" to classify the image and view the result.

Contributing:
        Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.


        THANK YOU......
        
