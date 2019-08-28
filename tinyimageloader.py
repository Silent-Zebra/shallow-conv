import os

import numpy
import scipy
import scipy.misc
import numpy as np

# paths to various data files
data_file_path = os.path.expanduser("~/tiny-images/tiny_images.bin")

# open data files
data_file = 0


def openTinyImage():
    global data_file
    data_file = open(data_file_path, "rb")


img_count = 79302017


def sliceToBin(indx):
    offset = indx * 3072
    data_file.seek(offset)
    data = data_file.read(3072)
    return numpy.fromstring(data, dtype='uint8')


def sliceToImage(data, path):
    t = data.reshape(32, 32, 3, order="F").copy()
    img = scipy.misc.toimage(t)
    img.save(path)


def closeTinyImage():
    data_file.close()




num_images = 800000

rand_nums = np.random.randint(0, img_count, num_images)

output_path = os.path.expanduser("~/tiny-images-subset/")

openTinyImage()

os.mkdir(output_path)
for i in range(num_images):
    sliceToImage(sliceToBin(rand_nums[i]), output_path+"/"+str(i)+".png")

closeTinyImage()
