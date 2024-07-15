# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 23:34:42 2023

@author: AnonSpirit
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense , Dropout,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50 , DenseNet121, MobileNet, Xception
from tensorflow.keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator

# # مقادیر ورودی برنامه:
train_data_dir = "H:/Datasets/olive_Classification-1/train"  # مسیر پوشه آموزش
validation_data_dir = "H:/Datasets/olive_Classification-1/valid" # مسیر پوشه ارزیابی
class_labels=['Ragham_1', 'Ragham_2', 'Ragham_3', 'Ragham_4', 'Ragham_5']
image_size = (670, 850)
batch_size = 4
epochs = 20


class DataGenerator(Sequence):
    def __init__(self, data_dir, image_size, batch_size, num_classes):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_paths, self.labels = self.load_data()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [tf.keras.preprocessing.image.load_img(path, target_size=self.image_size) for path in batch_paths]
        batch_images = [tf.keras.preprocessing.image.img_to_array(img) for img in batch_images]
        batch_images = np.array(batch_images)
        batch_labels = to_categorical(batch_labels, num_classes=self.num_classes)
        return batch_images, batch_labels

    def load_data(self):
        image_paths = []
        labels = []
        class_dirs = os.listdir(self.data_dir)
        for i, class_dir in enumerate(class_dirs):
            class_path = os.path.join(self.data_dir, class_dir)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image_paths.append(image_path)
                labels.append(i)
        return image_paths, labels




train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size[0], image_size[1]),
    batch_size=batch_size,
    class_mode='categorical')

# تعریف ژنراتور ارزیابی
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_size[0], image_size[1]),
    batch_size=batch_size,
    class_mode='categorical')

# ایجاد یک مدل توالی (Sequential)
model_sequntial = Sequential()
# لایه کانولوشنی
model_sequntial.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(image_size[0], image_size[1], 3)))
# لایه کانولوشنی
model_sequntial.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model_sequntial.add(MaxPool2D(pool_size=(2,2)))
model_sequntial.add(Dropout(0.25))
model_sequntial.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model_sequntial.add(MaxPool2D(pool_size=(2,2)))
model_sequntial.add(Dropout(0.25))
# تبدیل خروجی لایه‌های کانولوشنی به یک بردار توسط لایه فلت
model_sequntial.add(Flatten())
# لایه مخفی (پرسپترون)
model_sequntial.add(Dense(500, activation='relu'))
model_sequntial.add(Dropout(0.4))
model_sequntial.add(Dense(250, activation='relu'))
model_sequntial.add(Dropout(0.3))
# لایه خروجی
model_sequntial.add(Dense(len(class_labels), activation='softmax'))
# پایان ساختار مدل توالی
# آموزش مدل توالی
model_sequntial.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_sequential = model_sequntial.fit(train_generator,
                                         steps_per_epoch=len(train_generator),
                                         epochs=epochs,
                                         validation_data=val_generator,
                                         validation_steps=len(val_generator))






model_sequntial.save("models/test_model_sequential.h5")
print("Finish Sequential ---->")



# # ساختار مدل VGG16
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions_vgg = Dense(len(class_labels), activation='softmax')(x)
model_vgg = Model(inputs=base_model_vgg.input, outputs=predictions_vgg)


# # آموزش مدل VGG16
model_vgg.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_vgg = model_vgg.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator)
                            )
model_vgg.save("models/test_model_vgg.h5")
print("Finish VGG16 ---->")



# # ساختار مدل ResNet50
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions_resnet = Dense(len(class_labels), activation='softmax')(x)
model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)

# # آموزش مدل ResNet50
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_resnet = model_resnet.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator)
                                  )
model_resnet.save("models/test_model_resnet.h5")
print("Finish ResNet50 ---->")



# # ساخت مدل DenseNet
base_model_densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_densenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions_densenet = Dense(len(class_labels), activation='softmax')(x)
model_densenet = Model(inputs=base_model_densenet.input, outputs=predictions_densenet)

# # آموزش مدل DenseNet
model_densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_densenet = model_densenet.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator)
                                      )
model_densenet.save("models/test_model_densenet.h5")
print("Finish model_densenet--->")




# # ساخت مدل MobileNet
base_model_mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_mobilenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions_mobilenet = Dense(len(class_labels), activation='softmax')(x)
model_mobilenet = Model(inputs=base_model_mobilenet.input, outputs=predictions_mobilenet)


# # آموزش مدل MobileNet

model_mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_mobilenet = model_mobilenet.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator)
                                        )
model_mobilenet.save("models/test_model_mobilenet.h5")
print("Finish model_mobilenet--->")

# # ساخت مدل Xception
base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_xception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions_xception = Dense(len(class_labels), activation='softmax')(x)
model_xception = Model(inputs=base_model_xception.input, outputs=predictions_xception)



# # آموزش مدل Xception
model_xception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_xception = model_xception.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator)
                                      )
model_xception.save("models/test_model_xception.h5")
print("Finish model_xception--->")

# نمودار دقت 
plt.plot(history_sequential.history['accuracy'])
plt.plot(history_sequential.history['val_accuracy'])
plt.plot(history_vgg.history['accuracy'])
plt.plot(history_vgg.history['val_accuracy'])
plt.plot(history_resnet.history['accuracy'])
plt.plot(history_resnet.history['val_accuracy'])
plt.plot(history_densenet.history['accuracy'])
plt.plot(history_densenet.history['val_accuracy'])
plt.plot(history_mobilenet.history['accuracy'])
plt.plot(history_mobilenet.history['val_accuracy'])
plt.plot(history_xception.history['accuracy'])
plt.plot(history_xception.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Sequential Train', 'Sequential Validation', 'VGG16 Train', 'VGG16 Validation', 'ResNet50 Train', 'ResNet50 Validation','Densenet Train', 'Densenet Validation', 'Mobilenet Train', 'Mobilenet Validation', 'Xception Train', 'Xception Validation'], loc='lower right')
plt.show()
# # نمودار هزینه 
plt.plot(history_sequential.history['loss'])
plt.plot(history_sequential.history['val_loss'])
plt.plot(history_vgg.history['loss'])
plt.plot(history_vgg.history['val_loss'])
plt.plot(history_resnet.history['loss'])
plt.plot(history_resnet.history['val_loss'])
plt.plot(history_densenet.history['loss'])
plt.plot(history_densenet.history['val_loss'])
plt.plot(history_mobilenet.history['loss'])
plt.plot(history_mobilenet.history['val_loss'])
plt.plot(history_xception.history['loss'])
plt.plot(history_xception.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Sequential loss', 'Sequential Validation loss', 'VGG16 loss', 'VGG16 Validation loss', 'ResNet50 loss', 'ResNet50 Validation loss','Densenet loss', 'Densenet Validation loss', 'Mobilenet loss', 'Mobilenet Validation loss', 'Xception loss', 'Xception Validation loss'], loc='lower right')
plt.show()

