import array
import struct
import numpy
import os
import requests
import zipfile
from matplotlib import pyplot
import random

filename = 'mnist.zip'

def mnist_read(images_path, labels_path):
    labels = []
    with open(labels_path, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        labels = array.array('B', file.read())

    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        image_data = array.array('B', file.read())

    images = []
    for k in range(size):
        images.append([0] * rows * cols)
    for j in range(size):
        img = numpy.array(image_data[j * rows * cols:(j + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[j][:] = img

    return numpy.array(images), numpy.array(labels)

def mnist_load(train_image_path, train_label_path,
               test_image_path, test_label_path):
    return mnist_read(train_image_path, train_label_path), \
        mnist_read(test_image_path, test_label_path)

def parse():
    output_file = 'mnist.zip'
    if not os.path.exists(output_file):
        url = 'https://github.com/artinte/tiny-datasets/raw/develop/mnist.zip'
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as file:
                file.write(response.content)
        else:
            print('Download failed')
    if not os.path.exists(filename):
        print('Zip file not exist')
    else:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('temp/')
 
        filename_list = ['train-images-idx3-ubyte',
                      'train-labels-idx1-ubyte',
                      't10k-images-idx3-ubyte',
                      't10k-labels-idx1-ubyte']
        return mnist_load(*[os.path.join('temp/mnist', filename) for filename in filename_list])


if __name__ == '__main__':    
    (x_train, y_train), (x_test, y_test) = parse()

    images_show = []
    labels_show = []
    row = 5
    col = 3
    random.seed(100)
    for i in range(0, row * 2):
        r = random.randint(0, len(x_train))
        images_show.append(x_train[r])
        labels_show.append(str(r) + ' ' + str(y_train[r]))
    for i in range(0, row):
        r = random.randint(0, len(x_test))
        images_show.append(x_test[r])
        labels_show.append(str(r) + ' ' + str(y_test[r]))
    index = 1
    for image, label in zip(images_show, labels_show):
        ax = pyplot.subplot(col, row, index)
        ax.set_xticks([])  # remove x-axis ticks
        ax.set_yticks([])
        pyplot.imshow(image, cmap='gray')
        pyplot.title(label)
        index += 1
    pyplot.tight_layout()
    pyplot.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.01)
    pyplot.savefig('res/mnist_dataset_sample.png')
    pyplot.show()
