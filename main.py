import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

base_dir = '/home/allysonpfeil/devel/insert_path_here' #linux file path
train_dir = os.path.join(base_dir, 'train/') #folder names within path
test_dir = os.path.join(base_dir, 'test/') #folder names within path
img_size = (64, 64)
batches = 32
epochs = 50

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

model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')
