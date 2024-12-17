import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import os
import cv2
from transformers import AutoModelForImageClassification

def image_preprocessing(image_path):
    images=[]
    for i in image_path:
        # print(i)
        img = cv2.imread(i)
        # converting into grayscale
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # convrting into binaryimage
        _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        binary_image = cv2.resize(binary_image, (224, 224))
        # binary_image = np.expand_dims(binary_image, axis=-1)
        binary_image = cv2.merge([binary_image, binary_image, binary_image])
        binary_image = binary_image/255
        binary_image = torch.from_numpy(binary_image)
        images.append(binary_image)
    return images

def predict_image(image_path, model):
    preprocessed_img = image_preprocessing(image_path)
    images = torch.stack(preprocessed_img)
    images = images.permute(0, 3, 1, 2)
    predictions = model(images).logits.detach().numpy()
    return predictions


model = AutoModelForImageClassification.from_pretrained("models/vit-base-beans")

def struck_images():

    folder_path = 'images'
    images_path = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        images_path.append(file_path)
    # print()
    images_path.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
  
    # print(images_path)  
    
    # images_path = images_path[:2] 
    
    predictions = predict_image(images_path, model)

    not_struck =[]
    for i in range(len(predictions)):
        if predictions[i].argmax().item() == 0:
            not_struck.append(images_path[i])

    # print(not_struck)
    return not_struck


# struck_images()