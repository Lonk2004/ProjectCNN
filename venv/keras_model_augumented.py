#Imports, tensorflow, keras, sklearn utilities, and fits processing
import tensorflow as tf 
import os
import numpy as np
from keras import Sequential
from astropy.io import fits 
from tensorflow.keras.metrics import Recall 
from tensorflow.keras.layers import RandomTranslation
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

#Function call to store fits images as a numpy array. Enter noimages to balance classes. 
def load_images(path, noimages): 
    with os.scandir(path) as files:
        images = []
        nofiles = 0
        for file in files: 
            file_list = [file for file in files if not file.name.startswith('.')]
            selected_files = random.sample(file_list, min(noimages, len(file_list)))
            #Exeption handling if dataset contains non fits file
            for file in selected_files:
                try: 
                    image = fits.open(file)
                except:
                    continue
                data = image[0].data 
                data = data.astype(np.float32)
                data = np.expand_dims(data, axis=-1)
                data = tf.image.resize(data, (128, 128))
                #Normalise the data for the CNN
                data = data / np.max(data)
                images.append(data)
                nofiles += 1 
    return np.array(images)


#Load all images in directories
gcimages = load_images('/Users/jackskinner/Documents/3rd Year/Computer Science/astrodataset/astrodataset/outputdata/outputfits/fitsgcs',248)
gcimage_bins = np.ones(len(gcimages))
galaxyimages = load_images('/Users/jackskinner/Documents/3rd Year/Computer Science/astrodataset/astrodataset/outputdata/outputfits/galaxies',271)
galaxyimage_bins = np.zeros(len(galaxyimages))
wanghaloimages = load_images('/Users/jackskinner/Documents/3rd Year/Computer Science/astrodataset/astrodataset/outputdata/outputfits/wang_galaxies/wang001-225,275-500',56)
wanghaloimage_bins = np.zeros(len(wanghaloimages))
wangcentreimages = load_images('/Users/jackskinner/Documents/3rd Year/Computer Science/astrodataset/astrodataset/outputdata/outputfits/wang_galaxies/wang225-275',169)
wangcentreimage_bins = np.zeros(len(wangcentreimages))
#Load synthetic gcs. testsynthetic used to validate CNNs ability to recognise synthetic data.
syntheticgcimages = load_images('/Volumes/Backup Plus/Pandas_Data/clusters',248)
syntheticgcimage_bins = np.ones(len(syntheticgcimages))
testsyntheticgcimages = load_images('/Volumes/Backup Plus/Pandas_Data/clusters',500)
testsyntheticgcimage_bins = np.ones(len(testsyntheticgcimages))
# Combine images
images = np.concatenate((gcimages, galaxyimages,wanghaloimages, wangcentreimages, syntheticgcimages), axis=0)
print(len(images))

# Combine labels
image_bins = np.concatenate((gcimage_bins, galaxyimage_bins,syntheticgcimage_bins, wangcentreimage_bins, wanghaloimage_bins), axis=0)


#Use train_test_split to create training and testing data, with balanced image bins
X_train, X_temp, y_train, y_temp = train_test_split(
    images, image_bins, test_size=0.2, random_state=10, shuffle=True, stratify=image_bins
)

# Split the test data into validation and testing data. 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=16, shuffle=True, stratify=y_temp
)
#Define the CNNs batch size
batch_size = 36
# Define the ImageDataGenerator with horizontal shift and shear augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda img: tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), #rotations done by wang
    width_shift_range=0.05,  # 5% horizontal shift
    height_shift_range=0.05,  # 5% vertical shift
    horizontal_flip=True,  # Random horizontal flip
    vertical_flip=True,  # Random vertical flip
    fill_mode='nearest'
)

#Define the approximate no of training images desired post augumentation.
total_images = 9000
augmented_images = []
augmented_labels = []

# Number of batches required to reach the desired number of images
batches_needed = total_images // batch_size
val_batches_needed = batches_needed / 10
print(y_train)
# Create augmented images using the datagen.flow()
image_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)


# Loop over the generator and save the generated images
for i in range(batches_needed):
    img_batch, label_batch = next(image_generator) # Get the next batch of augmented images
    augmented_images.extend(img_batch)               # Add generated images to the list
    augmented_labels.extend(label_batch)  
     

# Convert lists to numpy arrays to be used in datasets
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)
expaneded_images = np.concatenate((augmented_images, images), axis=0)
expanded_image_bins = np.concatenate((augmented_labels, image_bins), axis=0)

#create training, valadation and testing dataset.
dataset = tf.data.Dataset.from_tensor_slices((expaneded_images, expanded_image_bins))
dataset = dataset.shuffle(buffer_size=len(expaneded_images))
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=len(X_val))
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#And dataset to validate efficacy recognising synthetics
syntetic_dataset = tf.data.Dataset.from_tensor_slices((testsyntheticgcimages, testsyntheticgcimage_bins))
syntetic_dataset = syntetic_dataset.batch(batch_size)
syntetic_dataset = syntetic_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
# Keras model with 2 conv layers, max pooling, 1 dense layer and dropout. 
height = 128
width = 128
model = Sequential([
    tf.keras.Input(shape=(height, width, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

#Compile the model using the Adam optimiser, and using binary_crossentropy loss to get a Recall metric.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy', 
    metrics=['accuracy', Recall()] 
)

model.fit(dataset, epochs=18, validation_data=val_dataset, class_weight = class_weights)

#evaluate model on unseen test data
test_loss, test_acc, test_recall = model.evaluate(test_dataset)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")

#Evalulate model on unseen synthetic data
test_loss, test_acc, test_recall = model.evaluate(syntetic_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Recall: {test_recall:.4f}")