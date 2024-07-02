# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import Sequential

# Get Location for DataSets
loc_data_test = "../data/Dataset)olive/Test"
loc_data_train = "../data/Dataset)olive/Train"

# Set Size Images (by PX)
img_w = 180
img_h = 180

itrateLearn = 20

def LOAD_DATASET(locateDB,w,h):
    return tf.keras.utils.image_dataset_from_directory(
    locateDB,
    shuffle = True,
    image_size=(w , h),
    batch_size=32,
    validation_split=False
    )
def PLOT_DATA(sizeFig , indexRange , data):
    plt.figure(figsize=(sizeFig))
    for img , labl in data:
        for i in range(indexRange):
            plt.subplot(3,3,i+1)
            # plt.xticks([])
            # plt.yticks([])
            plt.axis('off')
            plt.imshow(img[i].numpy().astype('uint8'))
            plt.title(class_name[labl[i]])
    plt.show()

class_name = ['O_1' , 'O_2' , 'O_3'  ,'O_4' , 'O_5' ]

Train_data = LOAD_DATASET(loc_data_train ,img_w,img_h) 
Test_data = LOAD_DATASET(loc_data_test ,img_w,img_h) 
# PLOT_DATA((10,10) , 9 , Test_data)

base_model = Sequential([
    layers.Rescaling(1 / 255),
    layers.Conv2D(16 , 3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32 , 3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64 , 3 , padding='same' , activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(5),
    layers.Dense(units=len(class_name))
    ])

base_model.compile(optimizer='adam' ,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
                    , metrics=['accuracy'])

hist_model = base_model.fit(Train_data , validation_data=Test_data , epochs=itrateLearn)
loss , acc = base_model.evaluate(Test_data)

epoch_range = range(itrateLearn)

plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.plot(epoch_range,hist_model.history['accuracy'] , label = "accuracy")
plt.plot(epoch_range,hist_model.history['val_accuracy'] , label = "validation Accuracy")
plt.subplot(1,2,2)
plt.plot(epoch_range,hist_model.history['loss'] , label = "loss")
plt.plot(epoch_range,hist_model.history['val_loss'] , label = "validation loss")
plt.show()

imageforTest = 'R5O_24.png'
loaded_img = tf.keras.utils.load_img(imageforTest,target_size=(img_w , img_h))
image_arr = tf.keras.utils.array_to_img(loaded_img)
image_bat = tf.expand_dims(image_arr , 0)

prdOlive = base_model.predict(image_bat)
score = tf.nn.softmax(prdOlive)

plt.figure()
plt.imshow(image_arr)
plt.show()

print("Olive is class : ", class_name[np.argmax(score)])

base_model.save("cnn_olive.keras")






