import tensorflow as tf

import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


def Localize(image, label):
    # for img in image.numpy():
    R = []
    for img in image:
        original = img.copy()
        thresh = cv2.threshold(cv2.convertScaleAbs(img),
                               0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(i) for i in contours]
        bboxes = sorted(bboxes, key=lambda x: x[0])
        size_of_boxes = len(bboxes)

        for i in range(size_of_boxes):
            x, y, w, h = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
            ROI = original[y:y + h, x:x + w]

        dim = (80, 80)
        if size_of_boxes > 0:
            ROI = cv2.resize(ROI, dim, interpolation=cv2.INTER_NEAREST)
            ROI = np.reshape(ROI, (80, 80, 1))
            R.append(ROI)
        else:
            ROI = cv2.resize(thresh, dim, interpolation=cv2.INTER_NEAREST)
            ROI = np.reshape(ROI, (80, 80, 1))
            R.append(ROI)

    return np.array(R), np.array(label)


def create_dataset(path, batch_size=1):
    ds = tf.keras.preprocessing.image_dataset_from_directory(path, color_mode='grayscale',
                                                             batch_size=1)
    ds = ds.map(lambda item1, item2: tf.numpy_function(Localize, [item1, item2], [tf.float32, tf.int32]))

    images = []
    labels = []
    for i, j in ds:
        images.append(i.numpy())
        labels.append(j.numpy())

    i_len = len(images)
    images = np.array(images).reshape((i_len, 80, 80, 1))
    labels = np.array(labels).reshape((i_len, -1))

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(batch_size)
    return ds


train_ds = create_dataset('./Sample_DT/train', 64)
val_ds = create_dataset('./Sample_DT/val', 1)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(53)
])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save('alpha.h5')
# model.load('alpha.h5')
