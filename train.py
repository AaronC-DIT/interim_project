import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from imutils import paths
import random
import cv2
import os

BATCH_SIZE = 3
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150

'''

PREPROCESSING


'''
dataset = "C:/Users/aaron/Documents/FYP/CUB_200_2011/small"    #change to be relative

imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(50)
random.shuffle(imagePaths)

data = []
labels = []

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = load_img(imagePath,target_size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    image = img_to_array(image)
    data.append(image)
 
    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split()
    labels.append(l)


data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
#labels = to_categorical(labels,5)
print(labels)
print(lb.classes_)
#print("[INFO] class labels:")
#mlbinarize = MultiLabelBinarizer()
#labels = mlbinarize.fit_transform(labels)



(train_X, test_X, train_Y, test_Y) = train_test_split(data,
    labels, test_size=0.2, random_state=50)

print(train_X.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam",
    metrics=["accuracy"])

model.summary()

#Augmentation Configuration
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True,
rotation_range=10,  
width_shift_range=0.1,  # move images horizontally 
height_shift_range=0.1, # move images vertically
fill_mode ='nearest')  #fill newly created pixels as a result of rotations

test_datagen = ImageDataGenerator(rescale=1./255)


model.fit_generator(
    train_datagen.flow(train_X, train_Y, batch_size=3),
    steps_per_epoch=261//BATCH_SIZE, #NB_TRAIN_IMG
    epochs=20,
    validation_data=(test_X,test_Y),
    validation_steps=261//BATCH_SIZE) #NB_VALID_IMG//BATCH_SIZE
print('Processing time:',(end - start)/60)
model.save_weights('cnn_baseline.h5')
plt.style.use("ggplot")
plt.figure()
N = 20
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
