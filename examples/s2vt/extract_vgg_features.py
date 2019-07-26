#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2

sys.path.append('../../python/')
import caffe

#### Addition for UCF-101
import skimage.transform

from glob import glob
from skvideo.io import vread
####

class FeatureExtractor():
  def __init__(self,  weights_path, image_net_proto, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    #load model into object net
    # first parameter = path ptototxt file, second parameter = path caffemodel file, thrid parameter caffe.TEST
    self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    assert channel_mean.shape[0] == len(channel_mean_values)
    for channel_index, mean_val in enumerate(channel_mean_values):
      channel_mean[channel_index, ...] = mean_val
    
    self.transformer.set_mean('data', channel_mean)
    self.transformer.set_channel_swap('data', (2, 1, 0)) # BGR
    self.transformer.set_transpose('data', (2, 0, 1))

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size,
        *self.image_net.blobs['data'].data.shape[1:])
  
  def preprocess_image(self, image, verbose=False):
    if type(image) in (str, unicode):
      image = plt.imread(image)
    crop_edge_ratio = (256. - 224.) / 256. / 2
    ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
    if verbose:
      print 'Preprocessed image has shape %s, range (%f, %f)' % \
          (preprocessed_image.shape,
           preprocessed_image.min(),
           preprocessed_image.max())
    return preprocessed_image

  def image_to_feature(self, image, output_name='fc7'):
    net = self.image_net
    if net.blobs['data'].data.shape[0] > 1:
      batch = np.zeros_like(net.blobs['data'].data)
      batch[0] = image
    else:
      batch = image.reshape(net.blobs['data'].data.shape)
    net.forward(data=batch)
    feature = net.blobs[output_name].data[0].copy()
    return feature

  def compute_features(self, image_list, output_name='fc7'):
    batch = np.zeros_like(self.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    features_shape = (len(image_list), ) + \
        self.image_net.blobs[output_name].data.shape[1:]
    features = np.zeros(features_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      for batch_index, video_path in enumerate(batch_list):
        ## Extract a frame in video
        images = extractFramesFromVideo(video_path)
        batch[batch_index:(batch_index + 1)] = self.preprocess_image(images[0])

      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      print 'Computing features for images %d-%d of %d' % \
          (batch_start_index, batch_start_index + current_batch_size - 1,
           len(image_list))
      self.image_net.forward(data=batch)
      features[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]

      if batch_start_index > 500:
        break
        
    return features


def extractFramesFromVideo(video_path):
  vid = vread(video_path)
  curr_frames = []
  for frame in vid:
      #frame = skimage.transform.resize(frame,[224,224])
      #if len(frame.shape)<3:
      #    frame = np.repeat(frame,3).reshape([224,224,3])
      curr_frames.append(frame)
  curr_frames = np.array(curr_frames)
  print ("Shape of frames: {0}".format(curr_frames.shape))
  maxFrames = 80 if curr_frames.shape[0] >= 80 else int(curr_frames.shape[0] - 2)
  idx = map(int,np.linspace(0,len(curr_frames)-1, maxFrames ))
  idx = list(idx)

  for indexToIterate, indexToExtract in enumerate(idx):
      curr_frames[indexToIterate] = curr_frames[indexToExtract]
      if indexToIterate == len(idx):
          curr_frames = curr_frames[0: indexToIterate]

  curr_frames = curr_frames[idx,:,:,:]
  return curr_frames


def write_features_to_file(image_list, features, output_file):
  with open(output_file, 'w') as opfd:
    for i, image_path in enumerate(image_list):
      image_feature = features[i].tolist()
      text_features = ','.join(map(str, image_feature))
      opfd.write('%s,%s\n' % (image_list[i], text_features))

def compute_single_image_feature(feature_extractor, image_path, out_file):
  assert os.path.exists(image_path)
  preprocessed_image = feature_extractor.preprocess_image(image_path)
  feature = feature_extractor.image_to_feature(preprocessed_image)
  write_features_to_file([image_path], [feature], out_file)

def compute_image_list_features(feature_extractor, images_file_path, out_file):
  assert os.path.exists(images_file_path[1])    #Check only for one
  image_list = images_file_path                 #Changed implementation
  features = feature_extractor.compute_features(image_list)
  write_features_to_file(image_list, features, out_file)
'''
def video_frame(video_name, image_name, videoList):
  counter = 1
  listVid = video_name
  for vid in listVid:
      vid = video_name + vid
      cap = cv2.VideoCapture(vid) #pass video's path
      try:
          if not os.path.exists('video_frame'):
              os.makedirs('video_frame')
      except OSError:
          print ('Error: Creating directory of video_frame')
      i=0
      counter+=1
      while(True):
          ret, frame = cap.read()
          if ret:
              name = './video_frame/'+image_name+'_'+str(i)+'.jpg'
              print('Creating...'+name)
              cv2.imwrite(name,frame)
              i+=1
          else:
              break


      cap.release()
      cv2.destroyAllWindows()
 

'''
def main():
  BASE_DIR = ''
  
  '''Populate IMAGE_LIST_FILE with all of the videos to convert'''
  BASE_IMAGE_LIST_FILE = os.path.dirname(os.path.realpath(__file__))
  
  IMAGE_NET_FILE = BASE_IMAGE_LIST_FILE
  MODEL_FILE = os.path.join(BASE_IMAGE_LIST_FILE, "snapshots")
  
  BASE_IMAGE_LIST_FILE = os.path.split(os.path.split(os.path.split(BASE_IMAGE_LIST_FILE)[0])[0])[0]
  BASE_IMAGE_LIST_FILE = os.path.join(BASE_IMAGE_LIST_FILE, "mocogan", "raw_data")
  
  if (os.path.exists(os.path.join(BASE_IMAGE_LIST_FILE, "UCF-101"))):
    IMAGE_CLASSES = glob(os.path.join(BASE_IMAGE_LIST_FILE, "UCF-101", "*"))
    VIDEO_PATH = glob(os.path.join(BASE_IMAGE_LIST_FILE, "UCF-101", "*","*"))
    
  else:
    IMAGE_CLASSES = glob(os.path.join(BASE_IMAGE_LIST_FILE, "*"))
    VIDEO_PATH = glob(os.path.join(BASE_IMAGE_LIST_FILE, "*","*"))  

  imageClasses = []     # This will be populated with Image Classes, this file is for output of Dataset
  for image_path in IMAGE_CLASSES:
      className = os.path.split(image_path)[1]
      imageClasses.append(className)
        
  imageClasses = list(dict.fromkeys(imageClasses))
  
  OUTPUT_FILE = 'output_features.csv'
  BATCH_SIZE = 10
  #VIDEO_LIST_PATH = "featureVideo.txt"
  # NOTE: Download these files from the Caffe Model Zoo.
  #IMAGE_NET_FILE = '../../models/vgg/vgg_orig_16layer.deploy.prototxt'
  #MODEL_FILE = MODEL_FILE + 'Nets/vgg/VGG_ILSVRC_16_layers.caffemodel'


  #IMAGE_NET_FILE = os.path.join(IMAGE_NET_FILE, 's2vt.words_to_preds.deploy.prototxt')
  #MODEL_FILE = os.path.join(MODEL_FILE, 's2vt_vgg_rgb.caffemodel')
  IMAGE_NET_FILE = BASE_DIR+ 'vgg.deploy.prototxt'
  MODEL_FILE = BASE_DIR+ 'snapshots/s2vt_vgg_rgb.caffemodel'  
  DEVICE_ID = -1

  feature_extractor = FeatureExtractor(MODEL_FILE, IMAGE_NET_FILE, DEVICE_ID)
  feature_extractor.set_image_batch_size(BATCH_SIZE)

  # compute features for a list of images in a file
  compute_image_list_features(feature_extractor, VIDEO_PATH, OUTPUT_FILE)
  # compute features for a single image
  # feature_extractor.set_image_batch_size(1)
  # compute_single_image_feature(feature_extractor, IMAGE_PATH, OUTPUT_FILE)

if __name__=="__main__":
  main()
