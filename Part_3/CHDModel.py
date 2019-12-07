from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

print("--Get data--")
train_data_path = 'heart_train.csv'
test_data_path = 'heart_test.csv'

df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

print("--Process data--")
df_train['famhist'] = pd.Categorical(df_train['famhist'])
df_train['famhist'] = df_train.famhist.cat.codes

df_test['famhist'] = pd.Categorical(df_test['famhist'])
df_test['famhist'] = df_test.famhist.cat.codes



target_train = df_train.pop('chd')
target_test = df_test.pop('chd')

dataset_train = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))
dataset_test = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))

tf.constant(df_train['famhist'])
tf.constant(df_test['famhist'])

train_dataset = dataset_train.shuffle(len(df_train)).batch(1)
test_dataset = dataset_test.shuffle(len(df_test)).batch(1)

def get_compiled_model():
  print("--Make model--")
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(500,kernel_regularizer=regularizers.l2(0.01), activation='relu'),
	tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(250, kernel_regularizer=regularizers.l2(0.01),activation='relu'),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Dense(125, kernel_regularizer=regularizers.l2(0.01),activation='relu'),
	tf.keras.layers.Dropout(0.25),
	tf.keras.layers.Dense(50, kernel_regularizer=regularizers.l2(0.01),activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
  return model

print("--Fit model--")
model = get_compiled_model()
model.fit(train_dataset, epochs=10)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(test_dataset)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")