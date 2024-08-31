# Import Libraries
import tensorflow as tf
from keras import layers , Sequential ,Model
from keras.api.applications.xception import Xception
# Get Location for DataSets
loc_data_test = "../data/test"
loc_data_train = "../data/train"

# Set Size Images (by pixel)
img_w = 180
img_h = 180

iterateLearn = 20

def LOAD_DATASET(locateDB,w,h):
    return tf.keras.utils.image_dataset_from_directory(
    locateDB,
    shuffle = True,
    image_size=(w , h),
    batch_size=32,
    validation_split=False)

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

base_model_xcp = Xception(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_xcp.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_xception = layers.Dense(len(class_name), activation='softmax')(x)
model_xcp = Model(inputs=base_model_xcp.input, outputs=predictions_xception)
# y_train_categorical = to_categorical(y_train, num_classes=class_name)  # Reshape (optional)
model_xcp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_xcp.fit(Train_data ,predictions_xception , epochs=20)
base_model.save("XCP.keras")
print("Finish model_xception--->")

base_model.compile(optimizer='adam' ,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
                    , metrics=['f1-score'])

hist_model = base_model.fit(Train_data , epochs=20)
loss , acc = base_model.evaluate(Test_data)
base_model.save("cnn_olive.keras")





