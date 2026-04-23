import numpy as np
import tensorflow as tf
from tensorflow import keras

#load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
#reshape and flatten the data
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

#one-hot
y_train_one_hot = tf.one_hot(y_train, 10)
y_test_one_hot = tf.one_hot(y_test, 10)

#models

def make_models(architecture):
  return keras.Sequential([keras.layers.Dense(units, activation = act)
                           for units, act in architecture] + [keras.layers.Dense(10, activation = "softmax")])
  
  
models = {
  "linear": make_models([]),
  "Dnn": make_models([(512, "relu"), (256, "relu"), (128, "relu")]),
  "Deeper_dnn": make_models([(784, "relu"), (512, "relu"), (512, "relu"), (256, "relu"), (256, "relu"), (128, "relu")]),
}

#Train and evaluate
def train_evaluate(name, model):
  print(f"Model name{name}")
  model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=["accuracy"])
  model.fit(X_train, y_train_one_hot, epochs = 10, batch_size = 256, verbose = 1)
  
  pred = model.predict(X_test, verbose = 0).argmax(axis=1)
  labels = y_test
  acc = (pred == labels).mean() * 100
  print(f"Test accuracy: {acc:}%")
  np.savez(f"output_{name}.npz", pred = pred, labels = labels)
  return pred, labels


#Overconfidence analysis

def overconfidence(pred, labels, c):
  mask = pred == c
  precision = (labels[mask] == c).mean() if mask.any() else float("NAN")
  print(f"Class {c}: {mask.sum()} predicted, {(labels[mask]==c).sum()} correct -> precison = {precision*100:.2f}%")
  
  return precision

Class_C = 3

for name, model, in models.items():
  pred, labels = train_evaluate(name, model)
  overconfidence(pred, labels, Class_C)
  