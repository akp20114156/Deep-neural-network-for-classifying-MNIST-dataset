from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#upload data
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#define output that the network should produce
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# image augumentation 
# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        #featurewise_center=True,  # set input mean to 0 over the dataset
        #samplewise_center=True,  # set each sample mean to 0
        #featurewise_std_normalization=True,  # divide inputs by std of the dataset
        #samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
train_gen = datagen.flow(x_train, y_train, batch_size=64)
test_gen = datagen.flow(x_test, y_test, batch_size=64)
print(len(train_gen))
# building cnn model - 
#nets = 10
#net = [0] *nets
#for j in range(nets):

net = Sequential()
net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding ='same', input_shape=(28,28,1)))
net.add(BatchNormalization())
net.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding ='same'))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
#net.add(Dropout(rate=0.4))

#net.add(BatchNormalization())
net.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
net.add(BatchNormalization())
net.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
#net.add(Dropout(rate=0.4))

#net.add(BatchNormalization())
net.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
net.add(BatchNormalization())
net.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
#net.add(Dropout(rate=0.4))

net.add(Flatten())
#net.add(BatchNormalization())
net.add(Dense(256, activation='relu'))
net.add(Dropout(rate=0.4))
net.add(Dense(128,activation="relu"))
net.add(Dense(10, activation='softmax'))
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


## training model
#history = net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=256)
#history = [0] * nets
#epochs = 50

history = net.fit(train_gen, epochs=20, steps_per_epoch=x_train.shape[0]//64, validation_data=test_gen, validation_steps=x_test.shape[0]//64)
net.save("network_for_mnist.h5")

# testing model
outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)
print('Accuracy = ',100-100*misclassified/labels_test.size)
