#run in the directory containing separate folders for each bird type

import numpy
from scipy.misc import imread, imresize, imsave
import cPickle, gzip, sys, os.path
folders = os.listdir(".")
features = ( )
for foldername in folders:
  if foldersname == ".DS_Store":
    continue
  files = os.listdir(foldername)
    for file in files:
      if file == ".DS_Store":
        continue
      img = imread(foldername + "/" + file)
      img = imresize(img, (128, 128))
      print "Adding image " + file + " with dimensions " + str(img.shape)
      features = features + (img,)


features = numpy.array(list(features))
idx = 0
for image in features:
  imsave("../128/bird" + str(idx) + ".png", image)	
  idx = idx + 1
  print "Saving" + str(idx - 1)
