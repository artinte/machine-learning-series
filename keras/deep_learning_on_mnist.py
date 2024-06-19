import keras
import numpy

epochs = 5

if __name__ == '__main__':
    print('Keras version: ', keras.__version__)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('Shape of train image:', x_train.shape)
    print('Shape of test image:', x_test.shape)

    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)

    model.evaluate(x_test, y_test, verbose=2)

    predictions = numpy.argmax(model.predict(x_test[:5]), axis=1)
    print("Predictions: " + str(predictions), ", Labels: " + str(y_test[:5]))