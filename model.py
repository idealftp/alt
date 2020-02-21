from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

# structure #AlexNet
# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
#classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
#classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
#classifier.add(BatchNormalization())

# Convolution Step 3
#classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
#classifier.add(BatchNormalization())

# Convolution Step 4
#classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
#classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
#classifier.add(Dense(units = 4096, activation = 'relu'))
#classifier.add(Dropout(0.4))
#classifier.add(BatchNormalization())
#classifier.add(Dense(units = 4096, activation = 'relu'))
#classifier.add(Dropout(0.4))
#classifier.add(BatchNormalization())
#classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
#classifier.add(BatchNormalization())
classifier.add(Dense(units = 2, activation = 'softmax'))

classifier.summary()

# Compiling the CNN
classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])



# image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rotation_range=40,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

train_data_dir = "../data/train"     # directory of training data

test_data_dir = "../data/test"      # directory of test data

#valid_data_dir = "../data/validation"      # directory of validation data

#rehefa avy eo manazava azy hoe manino mampiasa cross-validation

training_set = train_datagen.flow_from_directory(train_data_dir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_data_dir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')

print(training_set.class_indices)
""" 
history = classifier.fit_generator(training_set,
                                   steps_per_epoch=training_set.samples//batch_size,
                                   validation_data=test_set,
                                   epochs=500,
                                   validation_steps=test_set.samples//batch_size) """

#saving the model
filepath="model.hdf5"
classifier.save(filepath)
classifier.load_weights(filepath)

import cv2

#Image desease
img_dss = cv2.imread("‪D:/ideal/data/validation/disease/disease_02.PNG")
cv2.imshow("predict",img_dss)
cv2.waitKey()
img_dss = cv2.resize(img_dss, (224, 224)) 

print(classifier.predict(img_dss))

""" #plot stats : training

sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()






 """