from tensorflow import keras

# Load the saved model for prediction
loaded_model = keras.models.load_model('/content/drive/My Drive/Side_Projects/Image_Classification/efficientnetv2.h5')

# Predict an image using the loaded model
img_path = '/content/drive/My Drive/Side_Projects/Image_Classification/dataset/rain/35.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
prediction = loaded_model.predict(img_array)
predicted_class = keras.applications.imagenet_utils.decode_predictions(prediction, top=1)[0][0][1]
print('Predicted class:', predicted_class)