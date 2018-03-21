# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

# d = 64                           # dimension
# nb = 100000                      # database size
# nq = 10000                       # nb of queries
# np.random.seed(1234)             # make reproducible
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.

# import numpy as np
# import imageio
# import glob
# images = np.empty((0, 10000))
#



#  NOT SURE WHY WHEN READING THE FILE I AM NOT GETTING FLOAT TYPES





# for filename in glob.iglob('temp/output/*.JPG'):
#     i = imageio.imread(filename)
#     print(i)
#     file_array = np.matrix(i)
#     file_array = file_array.flatten()
#     # print(images.shape)
#     # print(file_array.shape)
#     images = np.append(images, file_array, axis=0)
#
# print(images.shape)
# print(images)
#
#
#
# import faiss                   # make faiss available
# index = faiss.IndexFlatL2(10000)   # build the index
# print(index.is_trained)
# print(images.shape)
# index.add(images)                  # add vectors to the index
# print(index.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I)
# print(D)
# D, I = index.search(xq, k)     # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries


# import imageio
# from skimage import color
# from skimage.transform import rescale, resize, downscale_local_mean
# import glob
# import os
#
#
#
# for filename in glob.iglob('temp/*.JPG'):
#     im = imageio.imread(filename)
#     image_resized = resize(im, (100, 100))
#     image = color.rgb2gray(image_resized)
#     imageio.imwrite(f'temp/output/{os.path.basename(filename)}', image)
#     print(image)


import imageio
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean
import glob
import faiss

images = np.empty((0, 10000))
for filename in glob.iglob('temp/*.JPG'):
    im = imageio.imread(filename)
    image_resized = resize(im, (100, 100))
    file_array = np.matrix(color.rgb2gray(image_resized))
    file_array = file_array.flatten()
    print(file_array)
    images = np.append(images, file_array, axis=0)

print(images.shape)
print(images)

index = faiss.IndexFlatL2(10000)   # build the index
print(index.is_trained)
print(images.shape)
index.add(images)                  # add vectors to the index
print(index.ntotal)