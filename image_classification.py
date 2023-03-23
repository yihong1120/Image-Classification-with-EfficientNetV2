import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the EfficientNetV2 model
model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation='softmax',
    include_preprocessing=True
)

# Add a new global average pooling layer followed by a dense layer with sigmoid activation
x = keras.layers.GlobalAveragePooling2D()(model.output)
x = keras.layers.Dense(1, activation='sigmoid')(x)

# Build the new model
model = keras.models.Model(inputs=model.inputs, outputs=x)

# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(10)
# ])

# Set the data directory path
data_dir = '/content/drive/My Drive/Side_Projects/Image_Classification/dataset/'

# Create an ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1./255, # Rescale pixel values between 0 and 1
    rotation_range=20, # Rotate images randomly within 20 degrees
    width_shift_range=0.2, # Randomly shift images horizontally within 20% of the image width
    height_shift_range=0.2, # Randomly shift images vertically within 20% of the image height
    shear_range=0.2, # Randomly apply shearing transformations within 20 degrees
    zoom_range=0.2, # Randomly zoom in on images within 20% of the original size
    horizontal_flip=True, # Flip images horizontally
    fill_mode='nearest', # Fill missing pixels with nearest pixel values
    validation_split=0.2 # Split 20% of data as validation set
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Use flow_from_directory function to create training and testing datasets
train_dataset = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training' # Specify as training set
)

val_dataset = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation' # Specify as validation set
)

test_dataset = test_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False # Do not shuffle, for evaluation purpose
)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save the model
model.save('/content/drive/My Drive/Side_Projects/Image_Classification/efficientnetv2.h5')

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy:', accuracy)