import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.optimizers import Adamax
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

dataPath = 'dataset'
classes = []
class_paths = []
# find folders in dataPath
files = os.listdir(dataPath)
for file in files:
    # make class paths
    label_dir = os.path.join(dataPath, file)
    # get image names
    label = os.listdir(label_dir)
    for image in label:
        # combine class paths with img paths
        image_path = os.path.join(label_dir, image)
        class_paths.append(image_path)
        # store folder names as classes
        classes.append(file)

# make a class for every img
image_classes = pd.Series(classes, name='Class')
# make a path for every img
image_paths = pd.Series(class_paths, name='Class Path')
#combine every img path with its class
train_dataframe = pd.concat([image_paths, image_classes], axis=1)

# split the train data and store the rest in "_"
train_data, _ = train_test_split(train_dataframe, test_size=.3, shuffle=True, random_state=20)
# split the rest data from "_"
val_data, ts_data = train_test_split(_, test_size=.5, shuffle=True, random_state=20)

# parameters
batch_size = 16
img_size = (224, 224)
channels = 3 #rgb
img_shape = (224, 224, 3)

# data augmentation using ImageDataGenerator
gen = ImageDataGenerator(rotation_range = 30)
test_gen =ImageDataGenerator()

train_gen = gen.flow_from_dataframe(train_data, x_col='Class Path', y_col='Class'
                                    , target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)
val_gen = gen.flow_from_dataframe(val_data, x_col= 'Class Path', y_col= 'Class',
                                   target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)
ts_gen = test_gen.flow_from_dataframe(ts_data, x_col= 'Class Path', y_col= 'Class',
                                   target_size= img_size, class_mode= 'categorical',
                                    color_mode= 'rgb', shuffle= True, batch_size= batch_size)

# make  a dic for classes 'Cloudy': 0, 'Rain': 1, 'Shine': 2, 'Sunrise': 3
g_dict = train_gen.class_indices
# list the classes from dic
classes = list(g_dict.keys())

base_model = tf.keras.applications.InceptionV3(input_shape=img_shape, include_top=False, weights = "imagenet")
base_model.trainable = False

model_inc = tf.keras.Sequential([ base_model,
                                  tf.keras.layers.MaxPooling2D(),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(4, activation="softmax")])

model_inc.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
epochs = 10

# Checkpoint callback to save best model weights
checkpoint = ModelCheckpoint(
    filepath=f"saved_models/InceptionV3_checkpoint.hdf5",
    monitor='val_accuracy',  # Monitor validation accuracy
    verbose=1,
    save_best_only=True,  # Save only the best model
    save_weights_only=True,  # Save only weights, not entire model
    mode='max'  # Maximize validation accuracy
)

# Early stopping callback to halt training if validation accuracy plateaus
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=2,  # Stop after 5 epochs with no improvement
    mode='max',  # Maximize validation accuracy
    restore_best_weights=True  # Load best weights when stopping
)

# Fit the model with callbacks
history = model_inc.fit(
    x=train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=val_gen,
    shuffle=False,
    callbacks=[checkpoint, early_stopping]  # Include the callbacks
)

train_score = model_inc.evaluate(train_gen, verbose= 1)
val_score = model_inc.evaluate(val_gen, verbose= 1)
test_score = model_inc.evaluate(ts_gen, verbose= 1)

# print model validation values
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Val Loss: ", val_score[0])
print("Val Accuracy: ", val_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

Epochs = [i+1 for i in range(len(tr_acc))]
loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

# Plot training history
plt.figure(figsize= (20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout
plt.show()

preds = model_inc.predict_generator(ts_gen)
y_pred = np.argmax(preds, axis=1)

g_dict = ts_gen.class_indices
classes = list(g_dict.keys())

cm = confusion_matrix(ts_gen.classes, y_pred)
plt.figure(figsize= (10, 10))
plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation= 45)
plt.yticks(tick_marks, classes)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()

print(classification_report(ts_gen.classes, y_pred, target_names= classes))