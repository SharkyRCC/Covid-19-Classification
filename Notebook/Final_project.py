import os
import cv2
import matplotlib.pyplot as plt

len(os.listdir('COVID-19_Radiography_Dataset/COVID/images'))
len(os.listdir('COVID-19_Radiography_Dataset/Normal/images'))

img = cv2.imread('COVID-19_Radiography_Dataset/Normal/images/Normal-10005.png')

plt.imshow(img)
plt.show()

img.shape

import pandas as pd
import numpy as np

df = pd.read_excel('COVID-19_Radiography_Dataset/COVID.metadata.xlsx')
df.head()

urls = os.listdir('COVID-19_Radiography_Dataset/COVID/images')

path = "COVID-19_Radiography_Dataset/COVID/images/" + urls[0]
path

def loadImages(path, urls, target):
  images = []
  labels = []
  for i in range(len(urls)):
    img_path = path + "/" + urls[i]
    img = cv2.imread(img_path)
    img = img / 255.0
    #print(img_path)
    # if we want to resize the images
    img = cv2.resize(img, (100, 100))
    images.append(img)
    labels.append(target)
  images = np.asarray(images)
  return images, labels

covid_path = "COVID-19_Radiography_Dataset/COVID/images"
covidUrl = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidUrl, 1)

len(covidUrl), len(covidImages)

normal_path = "COVID-19_Radiography_Dataset/Normal/images"
normal_urls = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normal_urls, 0)

covidImages.shape
normalImages.shape

data = np.r_[covidImages, normalImages]
data.shape

targets = np.r_[covidTargets, normalTargets]
targets.shape

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, 3, input_shape=(100,100,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=32,epochs=5,validation_data=(x_test, y_test))

plt.plot(model.history.history['accuracy'], label = 'train accuracy')
plt.plot(model.history.history['val_accuracy'],label = 'test_accuracy')
plt.legend()
plt.show()

plt.plot(model.history.history['loss'], label = 'train loss')
plt.plot(model.history.history['val_loss'],label = 'test_loss')
plt.legend()
plt.show()

