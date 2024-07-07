# Import Libraries
import tensorflow as tf
from keras import layers
from keras import Sequential

# Get Location for DataSets
loc_data_test = "../Model/Data/Dataset)olive/Test"
loc_data_train = "../Model/Data/Dataset)olive/Train"

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
class_name = ['O_1' , 'O_2' , 'O_3'  ,'O_4' , 'O_5' ]

Train_data = LOAD_DATASET(loc_data_train ,img_w,img_h) 
Test_data = LOAD_DATASET(loc_data_test ,img_w,img_h) 

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

hist_model = base_model.fit(Train_data , epochs=itrateLearn)
loss , acc = base_model.evaluate(Test_data)
base_model.save("cnn_olive.h5")





