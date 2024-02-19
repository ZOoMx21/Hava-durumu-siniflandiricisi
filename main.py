import keras.losses
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.applications import VGG16
from keras.models import Model

# # Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)) (x)
x =MaxPooling2D()(x)
x =Conv2D(16, (3,3), 1, activation='relu')(x)
x =MaxPooling2D()(x)
x =Conv2D(16, (3,3), 1, activation='relu')(x)
x =MaxPooling2D()(x)
x =Flatten()(x)
x =Dense(256, activation='relu')(x)
x =Dense(4, activation='Softmax')(x)
predictions = Dense(2, activation='softmax')(x)





data = tf.keras.utils.image_dataset_from_directory('dataset', batch_size=32, validation_split=None, image_size=(256, 256)) #create data object with our images and classes
data_iterator = data.as_numpy_iterator() #read data as numpy matrix
batch = data_iterator.next()

#find out classes numbers
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# fig.savefig("plot.png")
# plt.show()

data = data.map(lambda x, y: (x/255, y)) #scale images from (0-255) to (0-1), x: features, y: classes
# print(len(data),"batches") #see how many batches been splitted

#decide batch number for train,validation and test
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

#store train, validation and test data in seperate objects
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
print(len(train), len(val), len(test))

# Create new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=10)



# #define model building api
# model = Sequential()
#
# #add layers to the model
# model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(4, activation='Softmax'))
#
# #compile the model
# model.compile('Nadam',loss=keras.losses.Hinge(), metrics=['accuracy'])
#
# logdir='logs'
#
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#
# hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

