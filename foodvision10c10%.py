# Food Vision -> 10 classes -> 10% data (data augmentation)
# https://www.kaggle.com/datasets/dansbecker/food-101

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from sklearn.metrics import confusion_matrix

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.efficientnet_v2 import EfficientNetV2B3

# folders
train_data = ImageDataGenerator(rescale=1/255).flow_from_directory(
    batch_size=32,
    shuffle=True,
    color_mode='rgb',
    target_size=(224, 224),
    class_mode='categorical',
    directory='path here'
)
test_data = ImageDataGenerator(rescale=1/255).flow_from_directory(
    batch_size=32,
    shuffle=False,
    color_mode='rgb',
    target_size=(224, 224),
    class_mode='categorical',
    directory='path here'
)

labels_text = list(train_data.class_indices.keys())

# modelling : 
augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomHeight(factor=0.1),
    tf.keras.layers.experimental.preprocessing.RandomWidth(factor=0.1),
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

base_model = EfficientNetV2B3(include_preprocessing=False, include_top=False)
base_model.trainable = False
input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
augmentation = augmentation_layers(input_layer)
x = base_model(augmentation, training=False)
pooling_layer = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(units=len(labels_text), activation='softmax')(pooling_layer)
efficient_net_v3_MODEL = tf.keras.Model(input_layer, output_layer)

base_model_layer = efficient_net_v3_MODEL.layers[1]
for layer in base_model_layer.layers[-12:]:
    layer.trainable = True

efficient_net_v3_MODEL.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

efficient_net_v3_MODEL.fit(
    x=train_data,
    verbose=2,
    epochs=10,
    shuffle=True
)

# confusion matrix
preds = efficient_net_v3_MODEL.predict(test_data)
preds_text = []
for pred in preds:
    preds_text.append(labels_text[np.argmax(pred)])
test_data_labels = test_data.labels
actuals = []
for each in test_data_labels:
    actuals.append(labels_text[each])
sns.heatmap(data=confusion_matrix(y_pred=preds_text, y_true=actuals), annot=True, xticklabels=labels_text, yticklabels=labels_text)
plt.show()
