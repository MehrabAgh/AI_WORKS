import tensorflow as tf 
from keras import Sequential , layers , Model , applications

class_name = ['O_1' , 'O_2' , 'O_3'  ,'O_4' , 'O_5' ]
img_w = 180
img_h = 180
itrateLearn = 20

def LOAD_DATASET(locateDB,w,h):
    return tf.keras.utils.image_dataset_from_directory(
    locateDB,
    shuffle = True,
    image_size=(w , h),
    batch_size=32,
    validation_split=False)

loc_data_test = "../Model/Data/Dataset)olive/Test"
loc_data_train = "../Model/Data/Dataset)olive/Train"
Train_data = LOAD_DATASET(loc_data_train ,img_w,img_h) 
Test_data = LOAD_DATASET(loc_data_test ,img_w,img_h) 

# ایجاد یک مدل توالی (Sequential)
model_sequntial = Sequential()
# لایه کانولوشنی
model_sequntial.add(layers.Conv2D(50, kernel_size=(3,3), strides=(1,1),
         padding='same', activation='relu', input_shape=(img_w, img_h, 3)))
model_sequntial.add(layers.Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model_sequntial.add(layers.MaxPool2D(pool_size=(2,2)))
model_sequntial.add(layers.Dropout(0.25))
model_sequntial.add(layers.Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model_sequntial.add(layers.MaxPool2D(pool_size=(2,2)))
model_sequntial.add(layers.Dropout(0.25))
# تبدیل خروجی لایه‌های کانولوشنی به یک بردار توسط لایه فلت
model_sequntial.add(layers.Flatten())
# لایه مخفی (پرسپترون)
model_sequntial.add(layers.Dense(500, activation='relu'))
model_sequntial.add(layers.Dropout(0.4))
model_sequntial.add(layers.Dense(250, activation='relu'))
model_sequntial.add(layers.Dropout(0.3))
# لایه خروجی
model_sequntial.add(layers.Dense(len(class_name), activation='softmax'))
# پایان ساختار مدل توالی
# آموزش مدل توالی
model_sequntial.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_sequential = model_sequntial.fit(Train_data,
                        steps_per_epoch=len(Train_data),
                        epochs=itrateLearn)                        

model_sequntial.save("models/test_model_sequential.keras")
print("Finish Sequential ---->")



# # ساختار مدل VGG16
base_model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_vgg.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions_vgg = layers.Dense(len(class_name), activation='softmax')(x)
model_vgg = Model(inputs=base_model_vgg.input, outputs=predictions_vgg)
# # آموزش مدل VGG16
model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_vgg = model_vgg.fit(Train_data,
                    steps_per_epoch=len(Train_data),
                    epochs=itrateLearn)
model_vgg.save("models/test_model_vgg.keras")
print("Finish VGG16 ---->")



# # ساختار مدل ResNet50
base_model_resnet = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_resnet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_resnet = layers.Dense(len(class_name), activation='softmax')(x)
model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
# # آموزش مدل ResNet50
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_resnet = model_resnet.fit(Train_data,
                    steps_per_epoch=len(Train_data),
                    epochs=itrateLearn)
model_resnet.save("models/test_model_resnet.keras")
print("Finish ResNet50 ---->")



# # ساخت مدل DenseNet
base_model_densenet = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_densenet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_densenet = layers.Dense(len(class_name), activation='softmax')(x)
model_densenet = Model(inputs=base_model_densenet.input, outputs=predictions_densenet)
# # آموزش مدل DenseNet
model_densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_densenet = model_densenet.fit(Train_data,
                    steps_per_epoch=len(Train_data),
                    epochs=itrateLearn)
model_densenet.save("models/test_model_densenet.keras")
print("Finish model_densenet--->")




# # ساخت مدل MobileNet
base_model_mobilenet =applications.MobileNet(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_mobilenet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_mobilenet = layers.Dense(len(class_name), activation='softmax')(x)
model_mobilenet = Model(inputs=base_model_mobilenet.input, outputs=predictions_mobilenet)
# # آموزش مدل MobileNet
model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_mobilenet = model_mobilenet.fit(Train_data,
                    steps_per_epoch=len(Train_data),
                    epochs=itrateLearn)
model_mobilenet.save("models/test_model_mobilenet.keras")
print("Finish model_mobilenet--->")


# # ساخت مدل Xception
base_model_xception = applications.Xception(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
x = base_model_xception.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_xception = layers.Dense(len(class_name), activation='softmax')(x)
model_xception = Model(inputs=base_model_xception.input, outputs=predictions_xception)
# # آموزش مدل Xception
model_xception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_xception = model_xception.fit(Train_data,
                    steps_per_epoch=len(Train_data),
                    epochs=itrateLearn)
model_xception.save("models/test_model_xception.keras")
print("Finish model_xception--->")