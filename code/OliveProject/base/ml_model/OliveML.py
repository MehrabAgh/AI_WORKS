# Import Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import layers , Sequential ,Model ,  utils , losses
from keras.api.applications import Xception , ResNet50 , VGG16

# Get Location for DataSets
loc_data_valid = "../data/validation"
loc_data_train = "../data/train"

# Set Size Images (by pixel)
img_w = 180
img_h = 180

iterateLearn = 20

def LOAD_DATASET(locateDB,w,h):
    return utils.image_dataset_from_directory(
    locateDB,
    shuffle = True,
    image_size=(w , h),
    batch_size=32,
    validation_split=False)

class_name = [1 , 2 , 3  ,4 , 5 ]

Train_data = LOAD_DATASET(loc_data_train ,img_w,img_h) 
Valid_data = LOAD_DATASET(loc_data_valid ,img_w,img_h) 

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
                    loss= losses.SparseCategoricalCrossentropy(from_logits=True) 
                    , metrics=['accuracy'])

hist_model = base_model.fit(Train_data , epochs=20)
# loss , acc = base_model.evaluate(validate_data)
base_model.save("cnn_olive.keras")

base_model_xcp = Xception(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_xcp.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions_xception = layers.Dense(len(class_name), activation='softmax')(x)
model_xcp = Model(inputs=base_model_xcp.input, outputs=predictions_xception)
model_xcp.compile(optimizer='adam', loss= losses.SparseCategoricalCrossentropy()  , metrics=['accuracy'])
model_xcp.fit(Train_data , epochs=20)
base_model_xcp.save("XCP.keras")
print("Finish model_xception--->")

#  this model after learning is high loss and low accuracy

# base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
# x = base_model_vgg.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(128, activation='relu')(x)
# predictions_vgg = layers.Dense(len(class_name), activation='softmax')(x)
# model_vgg = Model(inputs=base_model_vgg.input, outputs=predictions_vgg)
# model_vgg.compile(optimizer='adam',  loss= losses.SparseCategoricalCrossentropy() , metrics=['accuracy'])
# history_vgg = model_vgg.fit(Train_data,epochs=20)
# model_vgg.save("model_vgg.keras")
# print("Finish VGG16 ---->")



base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_resnet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions_resnet = layers.Dense(len(class_name), activation='softmax')(x)
model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
model_resnet.compile(optimizer='adam', loss= losses.SparseCategoricalCrossentropy() , metrics=['accuracy'])
history_resnet = model_resnet.fit(Train_data, epochs=20)
model_resnet.save("Resnet.keras")
print("Finish ResNet50 ---->")









