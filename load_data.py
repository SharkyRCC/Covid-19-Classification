import os
import cv2
import numpy as np

def load_images(path, urls, target):
    images = []
    labels = []
    for i in range(len(urls)):
        img_path = path + "/" + urls[i]
        img = cv2.imread(img_path)
        img = img / 255.0
        # if we want to resize the images
        img = cv2.resize(img, (100, 100))
        images.append(img)
        labels.append(target)
    images = np.asarray(images)
    return images, labels

def load_covid_data():
    covid_path = "data/COVID-19_Radiography_Dataset/COVID/images"
    covid_urls = os.listdir(covid_path)
    covid_images, covid_targets = load_images(covid_path, covid_urls, 1)
    return covid_images, covid_targets

def load_normal_data():
    normal_path = "data/COVID-19_Radiography_Dataset/Normal/images"
    normal_urls = os.listdir(normal_path)
    normal_images, normal_targets = load_images(normal_path, normal_urls, 0)
    return normal_images, normal_targets

def load_data():
    covid_images, covid_targets = load_covid_data()
    normal_images, normal_targets = load_normal_data()
    data = np.r_[covid_images, normal_images]
    targets = np.r_[covid_targets, normal_targets]
    return data, targets