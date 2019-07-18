#!/usr/bin/python
import sys
import cv2
import imageio
import pylab
import numpy as np
import caffe
import skimage.transform

from time import sleep

def extract_feats(filenames,batch_size):
    """Function to extract VGG-16 features for frames in a video.
       Input:
            filenames:  List of filenames of videos to be processes
            batch_size: Batch size for feature extraction
       Writes features in .npy files"""
    model_file = './model/VGG_ILSVRC_16_layers.caffemodel'
    deploy_file = './model/VGG_ILSVRC_16_layers_deploy.prototxt'
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_file,model_file,caffe.TEST)
    layer = 'fc7'
    #mean_file = './ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    #transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
    transformer.set_transpose('data',(2,0,1))
    transformer.set_raw_scale('data',255.0)
    net.blobs['data'].reshape(batch_size,3,224,224)

    saveDir = os.path.join(os.path.dirname(__file__), 'data')
    outputFilename = 'features_file.txt'
    fileToSave = os.path.join(saveDir, outputFilename)
    f = open(fileToSave, 'w'); f.close()

    print ("VGG Network loaded")
    #Read videos and extract features in batches
    for file in filenames:
        vid = imageio.get_reader(file,'ffmpeg')
        curr_frames = []
        for frame in vid:
            frame = skimage.transform.resize(frame,[224,224])
            if len(frame.shape)<3:
                frame = np.repeat(frame,3).reshape([224,224,3])
            curr_frames.append(frame)
        curr_frames = np.array(curr_frames)
        print ("Shape of frames: {0}".format(curr_frames.shape))
        maxFrames = 80 if curr_frames.shape[0] >= 80 else int(curr_frames.shape[0]/2)
        idx = map(int,np.linspace(0,len(curr_frames)-1, maxFrames ))
        idx = list(idx)

        for indexToIterate, indexToExtract in enumerate(idx):
            curr_frames[indexToIterate] = curr_frames[indexToExtract]
            if indexToIterate == len(idx):
                curr_frames = curr_frames[0: indexToIterate]

        curr_frames = curr_frames[idx,:,:,:]
        print ("Captured {2} frames of {0}: {1}".format(file, curr_frames.shape, maxFrames))
        curr_feats = []
        for i in range(0, maxFrames, batch_size):
            caffe_in = np.zeros([batch_size,3,224,224])
            curr_batch = curr_frames[i:i+batch_size,:,:,:]
            for j in range(batch_size):
                caffe_in[j] = transformer.preprocess('data',curr_batch[j])
            out = net.forward_all(blobs=[layer],**{'data':caffe_in})
            curr_feats.extend(out[layer])
            #print ("Appended {} features {}. {}".format(j+1,out[layer].shape,i))
            #sleep(1 + 1/2)
        
        fileNameToSave = os.path.split(file)[1]
        curr_feats = np.array(curr_feats)

        videoList = [ fileNameToSave[:-4] + f'_{index}' for index in range(curr_feats.shape[0]) ]
        with open(fileToSave, 'a') as opfd:
            for index, name in enumerate(videoList):
                frame_feature = curr_feats[index].tolist()
                text_features = ','.join(map(str, frame_feature))
                opfd.write('%s,%s\n' % (videoList[index], text_features))

        print ("Saved file {}\nExiting".format(fileToSave))
        #sleep(5)


if __name__ == '__main__':

    import os
    from glob import glob
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('--batch_size', default = 1, type = int, help= 'Set batch size to extract features from video.')
    args = args.parse_args()

    batch_size = args.batch_size

    text2videoGanDir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    mocoganDir = os.path.join(text2videoGanDir, 'mocogan')
    dataDir = os.path.join(mocoganDir, 'raw_data')

    videoNames =  sorted( glob(os.path.join(dataDir, '*', '*')) )

    extract_feats(videoNames, batch_size)