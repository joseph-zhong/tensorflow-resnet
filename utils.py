import numpy as np
import cv2
from synset import *
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    img = skimage.io.imread(path) # skimage will divide by 255.0 first!
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    plt.imshow(resized_img)
    plt.show()
    return resized_img

def load_cv_img(path, size=224):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img / 255.0 # using numpy arrays must divide by 255.0!
  assert (0 <= img).all() and (img <= 1.0).all()
  # print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
  # resize to 224, 224
  resized_img = cv2.resize(crop_img, (size, size))
  return resized_img

# returns the top1 string
def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print "Top1: %s | Confidence: %s " % (top1, prob[pred[0]])
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print "Top5: ", top5
    return top1