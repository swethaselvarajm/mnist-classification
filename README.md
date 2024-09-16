# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
![image](https://github.com/user-attachments/assets/666b2c1d-f802-4602-91e6-a782dd9e7e2f)

## Neural Network Model

![WhatsApp Image 2024-09-13 at 11 32 13_e1805954](https://github.com/user-attachments/assets/017b1d4a-9228-43c3-9e53-e75566da90ea)

## DESIGN STEPS

## STEP 1:
Import tensorflow and preprocessing libraries.

## STEP 2:
Download and load the dataset

## STEP 3:
Scale the dataset between it's min and max values

## STEP 4:
Using one hot encode, encode the categorical values

## STEP 5:
Split the data into train and test

## STEP 6:
Build the convolutional neural network model

## STEP 7:
Train the model with the training data

## STEP 8:
Plot the performance plot

## STEP 9:
Evaluate the model with the testing data

## STEP 10:
Fit the model and predict the single input


## PROGRAM

### Name:SWETHA S
### Register Number:212222230155

```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

X_train.shape

X_test.shape
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
print("SWETHA S, 212222230155")
single_image = X_train[400]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
print("SWETHA S, 212222230155")
metrics.head()
print("SWETHA S, 212222230155")
metrics[['accuracy','val_accuracy']].plot()
print("SWETHA S, 212222230155")
metrics[['loss','val_loss']].plot()
print("SWETHA S, 212222230155")
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("SWETHA S, 212222230155")
print(classification_report(y_test,x_test_predictions))

img = image.load_img('EX3PIC.jpg')
type(img)
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print('SWETHA S')
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![OUT1](https://github.com/user-attachments/assets/4e216929-3025-446b-984e-7f8df2e27202)

![OUT2](https://github.com/user-attachments/assets/caa11666-5dd4-4122-8cfe-c2f209e189e9)

![OUT3](https://github.com/user-attachments/assets/0e1df2d6-e9dd-4ae0-aab0-844b3f77f5e6)


### Classification Report

![OUT4](https://github.com/user-attachments/assets/c6963bdd-3439-4c7d-9ad3-0fe89ccf06e9)


### Confusion Matrix

![OUT6](https://github.com/user-attachments/assets/73288792-04df-436c-8587-0fc53e585dac)


### New Sample Data Prediction

![OUT5](https://github.com/user-attachments/assets/7420d4fb-5e32-4918-af85-334316eb94e9)


## RESULT
Thus a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.
