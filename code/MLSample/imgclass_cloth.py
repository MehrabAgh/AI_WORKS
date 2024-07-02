import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

all_clothes = tf.keras.datasets.fashion_mnist
(trainImg,trainLabel),(testImg,testLabel) = all_clothes.load_data()
# import data for input

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# store label name on list

def SHOW_CLOTH(i):
    plt.figure(figsize=(1,1))  
    plt.imshow(testImg[i])
    plt.xticks([])
    plt.yticks([]) 
    plt.grid(False)
    plt.show()
# draw cloth image 

trainImg = trainImg / 255.0
testImg = testImg / 255.0
# convert and normalize images between 0 - 1 

base_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # set input(Layer) data and transform on 28*28px
    tf.keras.layers.Dense(128 , activation='relu'), # this hidden layer and neuron connection
    tf.keras.layers.Dense(10) # this last layer for output data ( classes )
])
# create sequence for linear modeling nn

base_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 
# end settingup model nn ( optimizer , function loss , metrics ) 

base_model.fit(trainImg,trainLabel,epochs=10)
# learn model with training images

test_loss, test_acc = base_model.evaluate(testImg , testLabel , verbose='0')
# get result information for function loss & accuracy param on model

print('funcLoss : ', test_loss, 'accuracy : ',  test_acc) #show infos

probability_model = tf.keras.Sequential([base_model, tf.keras.layers.Softmax()])
# convert base model to probability model for samplize calculate 

prd_model =  probability_model.predict(testImg)
# predict model with test image

resultClass = np.argmax(prd_model[40])

print("Category :", class_names[resultClass])

# SHOW_CLOTH(40)




