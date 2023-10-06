# import packages
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Define data directories and parameters
base_dir = '/home/allysonpfeil/devel/actis xrays edit' #linux file path
train_dir = os.path.join(base_dir, 'train/') #folder names within path
test_dir = os.path.join(base_dir, 'test/') #folder names within path
img_size = (64, 64)
batches = 32
epochs = 50

# create data generators and augment training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255., #normalization
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255.) #normalization

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    class_mode='binary',
    batch_size=batches
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    class_mode='binary',
    batch_size=batches
)

# define the model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# define early stopping callback
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# end
