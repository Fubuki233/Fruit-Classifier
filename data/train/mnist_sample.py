import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image



'''
Show an sample digit.
'''
def show_sample_digit():
    img = Image.open('../../ml_data/mnist_train/0/img_1.jpg')
    plt.imshow(img, cmap='gray')
    plt.show()


'''
Performs onehot-encodings for every digit (0 to 9).
'''
def encode_onehot(pos, n_rows):
    # 10 classes (digit 0 to 9)
    y_onehot = [0] * 10
    # create onehot-encodings for digit (i - 1)
    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows
    # convert python list to numpy array
    # as keras requires numpy array
    return np.array(y_onehots)


'''
Read in all images in a directory.
'''
def read_img_data(path):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory
        img = Image.open("{}/{}".format(path, file))

        # convert image to numpy array
        data = np.array([np.asarray(img)])

        try:
            x_train = np.concatenate((x_train, data))
        except:
            x_train = data

    # image is 28x28 and since it is gray-scale, there
    # is only 1 channel which yields the shape (28, 28, 1).
    # reshape by using -1 to let numpy computes the number 
    # of rows with (28, 28, 1) as the shape of each row.
    return np.reshape(x_train, (-1, 28,28,1))     


'''
Read in all 10 folders - each folder contains images of a digit.
For example, /0 contains all images of digit 0, /1 contains all
images of digit 1, and so on.
'''
def prep_data(path):
    for i in range(10):
        data = read_img_data(path + str(i))

        try:
            x = np.concatenate((x, data))
        except:
            x = data       

        # construct the onehot-encodings for a digit's data
        y_onehots = encode_onehot(i, data.shape[0])
        try:
            y = np.concatenate((y, y_onehots))
        except:
            y = y_onehots           

    return x, y


'''
Create our model
'''
def create_model():
    model = tf.keras.Sequential()

    # create a CNN model
    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))    

    # output layer: performs classification; the image is either 0, 1, 
    # 2, 3, 4, 5, 6, 7, or 9 (10 possible classes)
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))    

    # build the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])

    return model


'''
Main program.
'''
def main():
    # create our CNN model
    model = create_model()

    # model architecture
    print(model.summary())

    # fetch train data and its associated onehot-encoded labels
    x_train, y_train = prep_data('../../ml_data/mnist_train/')
    
    # RGB values are from 0 to 255, by dividing by 255,
    # x_train is normalized to [0, 1].
    # train our model for 1 epoch
    model.fit(x=x_train/255, y=y_train, epochs=1)

    # showing how we can save our trained model
    model.save('../../ml_data/mnist_saved_model')
    
    # showing how we can load our trained model
    model = tf.keras.models.load_model('../../ml_data/mnist_saved_model')
    
    # fetch test data and its associated onehot-encoded labels
    x_test, y_test = prep_data('../../ml_data/mnist_test/')

    # test how well our model performs against data
    # that it has not seen before
    model.evaluate(x=x_test/255, y=y_test)


# running via "python mnist_sample.py"
if __name__ == '__main__':
  main()


