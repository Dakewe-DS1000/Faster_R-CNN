#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from lib.datasets.factory import get_imdb
import cv2
from lib.nets.vgg16 import vgg16
from lib.utils.test import im_detect
from lib.utils.test import test_net
from lib.config import config as cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    # Create session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    # init net
    net = vgg16(batch_size=1)
    net.create_architecture(sess, "TEST", 2, tag='default', anchor_scales=[8, 16, 32], anchor_ratios=(1, 2))

    for i in range(80001) :
        if i==0 : continue
        if (i%1000)!=0 : continue
        weights_filename = "default\\voc_2007_trainval\\default\\vgg16_faster_rcnn_iter_{0}.ckpt".format(i)
        print("Load weights : {0}".format(weights_filename))
        saver = tf.train.Saver()
        saver.restore(sess, weights_filename)
    #set imdb
        imdb_name = "voc_2007_test"
        imdb = get_imdb(imdb_name)
        accuracy, recall, precision, tpr, fpr = test_net(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.0)
        
        record_file = open("TestNet_Record_iter_{0}.csv".format(i), "w")
        record_file.write("accuracy,recall,precision,tpr,fpr\n")
        num = len(accuracy[0])
        print("Data Number = {0} :: {1}, {2}, {3}, {4}".format(num, len(recall[0]), len(precision[0]), len(tpr[0]), len(fpr[0])))
        for idx in range(num) :
            record_file.write("{0:3f},{1:3f},{2:3f},{3:3f},{4:3f}\n".format(accuracy[0][idx], recall[0][idx], precision[0][idx], tpr[0][idx], fpr[0][idx]))
        record_file.close()