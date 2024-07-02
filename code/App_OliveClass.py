import streamlit as sl
import tensorflow as tf
import numpy as np
import os
from PIL import Image

img_w = 180
img_h = 180

save_dir = "uploaded_images"

NN_Model = tf.keras.models.load_model("cnn_olive.keras")
Olive_Category = ['Type_1','Type_2','Type_3','Type_4','Type_5']

filenames = os.listdir(save_dir)
for file in filenames:
    print(file)
uploaded_file = sl.file_uploader("Choose a file", type=["jpg","jpeg", "png"])

if(uploaded_file):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fileName = uploaded_file.name
    save_path = os.path.join(save_dir, fileName)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
        sl.success(f"Image saved successfully to: {save_path}")
    saved_image = Image.open(save_path)
    
    inputImage = tf.keras.utils.load_img( save_path , target_size=(img_w , img_h))
    arrayImage = tf.keras.utils.array_to_img(inputImage)
    ImgAddDim = tf.expand_dims(arrayImage , axis=0)

    resultPred = NN_Model.predict(ImgAddDim) # type: ignore
    getScore = tf.nn.softmax(resultPred)

    sl.image(inputImage)
    sl.write("Olive is class : ", Olive_Category[np.argmax(resultPred)])
    sl.write("accuracy :" , np.max(getScore)*100,"%")    



