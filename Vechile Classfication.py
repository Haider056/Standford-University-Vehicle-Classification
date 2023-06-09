import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set the paths to your data and annotation files
train_path = '/content/drive/MyDrive/car_data/car_data/train'
test_path = '/content/drive/MyDrive/car_data/car_data/test'
anno_train_file = '/content/drive/MyDrive/anno_train.csv'
anno_test_file = '/content/drive/MyDrive/anno_test.csv'
names_file = '/content/drive/MyDrive/names.csv'

# Define the image dimensions
img_width, img_height = 224, 224

# Create the train and test data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the training data from the directory and apply data augmentation
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Determine the number of classes
num_classes = train_generator.num_classes

# Load the test data from the directory
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the InceptionV3 model (pre-trained on ImageNet)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 50  # You can adjust the number of epochs

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Save the trained model
model.save('/content/drive/MyDrive/car_classification_model2.h5')
