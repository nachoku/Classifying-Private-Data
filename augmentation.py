# Image Loading Code used for these examples
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)

    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

directory='/Users/nach/Downloads/zerr_dataset_cleaned/private/'

# img = Image.open(directory+'0a4c1a1cb7.jpg')
# WIDTH, HEIGHT = img.size
# img = np.array(img)
# img=random_noise(img)
# plt.imshow(img)
# plt.show()



# #Flipping
# flipped_img = np.fliplr(img)
# plt.imshow(flipped_img)
# plt.show()
#
# #Noise
#
# noise = np.random.randint(5, size = (164, 278, 4), dtype = 'uint8')
#
# for i in range(WIDTH):
#     for j in range(HEIGHT):
#         for k in range(DEPTH):
#             if (img[i][j][k] != 255):
#                 img[i][j][k] += noise[i][j][k]

directory='/Users/nach/Downloads/zerr_dataset_cleaned'

for class_name in os.listdir(directory):
    if not class_name.startswith('.'):
        print('Class Name: '+class_name)
        if class_name=='public':
           continue
        for image_name in os.listdir(directory+'/'+class_name):
            if not image_name.startswith('.'):
                #print(image_name)
                location=directory+'/'+class_name+'/'+image_name
                print(location)

                location=Image.open(location)
                location = np.array(location)
                # plt.imshow(location)
                # plt.show()
                try:
                    img=random_rotation(location)
                    sk.io.imsave(directory+'/transformed/RRotation_'+image_name, img)

                    img = random_noise(location)
                    sk.io.imsave(directory + '/transformed/RNoise_'+image_name, img)

                    img = horizontal_flip(location)
                    sk.io.imsave(directory + '/transformed/HFlip_'+image_name, img)
                except:
                    continue