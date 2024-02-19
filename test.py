from keras.models import load_model
from keras.preprocessing import image
import numpy as np

classes = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
model = load_model('saved_models/Xception.h5')
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(100, 100))  # Assuming Xception model input size is (299, 299)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0
prediction = model.predict(img_array)
class_index = np.argmax(prediction[0])
print(classes[class_index])
