import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data = tf.keras.utils.image_dataset_from_directory('dataset', batch_size=32, validation_split=None, label_mode='categorical') #create data object with our images and classes
data_iterator = data.as_numpy_iterator() #read data as numpy matrix
batch = data_iterator.next()
print(batch)