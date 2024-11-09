# In[1]:
import warnings
warnings.filterwarnings('ignore')

import os
import gc  # Importing garbage collector
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

# Enable memory growth for GPUs (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Image size
IMAGE_SIZE = [224, 224]

# Paths to datasets
train_path = r"#training path"
valid_path = r"#test path"

# Load the VGG16 model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the layers
for layer in vgg.layers:
    layer.trainable = False

# Get number of classes
folders = glob(train_path + '/*')

# Add fully connected layers
x = Flatten()(vgg.output)
# Use a single output node with sigmoid for binary classification
prediction = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# In[2]:

# Compile the model for binary classification
model.compile(
  loss='binary_crossentropy',  # Change to binary_crossentropy for binary classification
  optimizer='adam',
  metrics=['accuracy']
)

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and test data (class_mode='binary' for binary classification)
training_set = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=10, class_mode='binary')
test_set = test_datagen.flow_from_directory(valid_path, target_size=(224, 224), batch_size=10, class_mode='binary')

# Callbacks for early stopping and learning rate reduction
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# In[3]:

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),
    callbacks=[early_stopping, reduce_lr]
)



import tensorflow as tf
from keras.models import load_model
# save the model
model.save('my_model.keras')
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import numpy as np


# Load the saved model
model = load_model('my_model.keras')

# Load and preprocess a test image
img = image.load_img(r"#path of image in .jpeg format to get prediction", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# Predict the class of the image
classes = model.predict(img_data)
result = classes[0][0]

# Threshold for binary classification
if result < 0.5:
    print("Result is Normal")
else:
    print("Person is Affected By PNEUMONIA")
