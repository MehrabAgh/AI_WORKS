import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plot

img_w = 180
img_h = 180

def ProcessImage(img):
    if(img):            
        pathModel = os.path.join("/AI_WORKS/code/OliveProject/ML_Core/","cnn_olive.keras")
        NN_Model = tf.keras.models.load_model(pathModel)
        Olive_Category = ['Type_1','Type_2','Type_3','Type_4','Type_5']            
        inputImage = tf.keras.utils.load_img(img, target_size=(img_w , img_h))
        arrayImage = tf.keras.utils.array_to_img(inputImage)
        ImgAddDim = tf.expand_dims(arrayImage , axis=0)

        resultPred = NN_Model.predict(ImgAddDim) # type: ignore
        getScore = tf.nn.softmax(resultPred)      

        classname = Olive_Category[np.argmax(resultPred)]
        acc = np.max(getScore)*100
        # plot.figure(figsize=(10,10))
        # plot.plot(getScore ,np.argmax(resultPred) )
        # plot.show()

    return {'err':0 , 'acc':acc, 'className': classname}




