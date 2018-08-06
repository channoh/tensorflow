

import os
from urllib.request import urlopen
import gzip
import numpy

def maybe_download(filename, directory, url):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print("Download {} ...".format(filepath))
        data = urlopen(url).read()
        with open(filepath, 'wb') as f:
            f.write(data)
    return filepath

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

# def gunzip(in_path):
    # out_path = in_path.rsplit('.', 1)[0]
    # with gzip.open(in_path, 'rb') as in_file:
        # content = in_file.read()
        # with open(out_path, 'wb') as out_file:
            # out_file.write(content)


def extract_images(in_path):
    with gzip.open(in_path, 'rb') as in_file:
        magic = _read32(in_file)
        num_images = _read32(in_file)
        rows = _read32(in_file)
        cols = _read32(in_file)
        print(magic, num_images, rows, cols)
        buf = in_file.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

def extract_labels(in_path):
    with gzip.open(in_path, 'rb') as in_file:
        magic = _read32(in_file)
        num_items = _read32(in_file)
        print(magic, num_items)
        buf = in_file.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        num_classes = 10
        return dense_to_one_hot(labels, num_classes)
        # if one_hot:
            # return dense_to_one_hot(labels, num_classes)
        # return labels


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
