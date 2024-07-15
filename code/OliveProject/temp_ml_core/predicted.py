# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:43:43 2023
@author: AnonSpirit
"""

import os
import cv2
import numpy as np
from keras.models import load_model

# مسیر مدل‌های آموزش داده شده
model_dir = "models"
model_names = ["test_model_sequential.h5", "test_model_vgg.h5", "test_model_resnet.h5", "test_model_densenet.h5", "test_model_mobilenet.h5", "test_model_xception.h5"]

# مسیر تصاویر تست جدید
test_images_dir = "test"

# کلاس‌های مرتبط با زیتون
class_labels = ['Ragham_1', 'Ragham_2', 'Ragham_3', 'Ragham_4', 'Ragham_5']
image_size = (670, 850)

# بارگذاری مدل‌های آموزش داده شده
model = load_model(os.path.join(model_dir, model_names[4]))

def detect_olives(image):
    # پیش‌پردازش تصویر
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    # پیش‌بینی با استفاده از شبکه عصبی
    prediction = model.predict(image)
    # برچسب پیش‌بینی شده
    predicted_class_index = np.argmax(prediction)
    # احتمال پیش‌بینی
    probability = np.max(prediction) * 100
    return predicted_class_index, probability

# مسیر پوشه‌های Test و Train
test_dir = "H:/Datasets/olive_Classification-1/test"

# خروجی را در یک فایل متنی ذخیره می‌کنیم
output_file = "results.txt"

# باز کردن فایل برای نوشتن
with open(output_file, "w") as f:
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            class_index = os.path.basename(class_path)
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                # خواندن تصویر
                image = cv2.imread(image_path)
                # تشخیص کلاس زیتون در تصویر
                predicted_class_index, probability = detect_olives(image)
                # نوشتن نتیجه به فایل
                f.write('Image: {}\n'.format(image_path))
                f.write('Ground Truth Class: {}\n'.format(class_index))
                f.write('Predicted Class: {}\n'.format(class_labels[predicted_class_index]))
                f.write('Probability: {:.3f}%\n'.format(probability))
                f.write('\n')  # خط خالی بین نتایج مختلف

print("Results saved to:", output_file)
