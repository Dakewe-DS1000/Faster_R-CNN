#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
           #'Metastatics Lymph-node : ')

imgFilePath = "data\\demo\\src"  #Jpeg Images
modelName = "vgg16_faster_rcnn_iter_80000.ckpt"
NETS = {'vgg16': (modelName,), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, thresh=0.5, image_name=""):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox  = dets[i, :4] # Get bbox (x1, y1, x2, y2)
        score = dets[i, -1] # Get Score
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2

        ax.add_patch(
            #Draw Rectangle
            plt.Rectangle((bbox[0], bbox[1]), # Top Left Coordination : X = bbox[0], Y = bbox[1]
                          bbox[2] - bbox[0],  # Width  = bbox[2] - bbox[0]
                          bbox[3] - bbox[1],  # Height = bbox[3] - bbox[1]
                          fill=False,
                          edgecolor='red', linewidth=1.5) # color is Red and line width is 1.5 point
            #Draw Circle
            #plt.Circle((cx, cy), 4, color="red")
        )
        #print("Save image : {0}".format(imgName))
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.1f}% '.format(class_name, score*100.0),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=10, color='white')
        #lmImg = cv2.getRectSubPix(im, (bbox[2] - bbox[0], bbox[3] - bbox[1]), (bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2))
        #imgName = image_name.split("\\")[4]
        #imgName = imgName.split(".")[0]
        #imgName = "data\\demo\\dstA\\" + imgName+"-"+str(i)+".jpg"
        #cv2.imwrite(imgName, lmImg)

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name, thresh),
    #             fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if len(inds) == 0:
        return False
    else:
        return True


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im_file = os.path.join(imgFilePath, image_name)
    im = cv2.imread(im_file)
    #print("Read image file : {0}".format(im_file))

    # Detect all object classes and regress object bounds
    #timer = Timer()
    #timer.tic()
    scores, boxes = im_detect(sess, net, im)
    #timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1 #CONF Threshold = 0.1 Default 0.1
    NMS_THRESH = 0.0  #NMS Threshold  = 0.1 Default 0.1
    rtnVal = True
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        rtnVal = rtnVal & vis_detections(im, cls, dets, thresh=CONF_THRESH, image_name=im_file)
    return rtnVal


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    tfmodel = os.path.join("default", DATASETS[dataset][0], "default", NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    #tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,
                            tag='default', anchor_scales=[8, 16, 32], anchor_ratios=(1, 2))
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
      
    total_img   = os.listdir(imgFilePath)
    imgNumber   = len(total_img)
    failedNum   = 0
    succeedNum  = 0
    for idx in range(imgNumber-1) :
        im_name = total_img[idx]
        out_file = os.path.join(cfg.FLAGS2["data_dir"], "demo\\dst", im_name)
        if(demo(sess, net, im_name)==False):
            print("Failed to Detect")
            failedNum += 1
            #noDetect = cv2.imread(os.path.join(imgFilePath, im_name))
            #noDetect = cv2.resize(noDetect, (1200, 1200))
            #cv2.imwrite(out_file, noDetect)
            #continue
        out_file = os.path.join(cfg.FLAGS2["data_dir"], "demo\\dst", im_name)
        plt.savefig(out_file, dpi=300)
        plt.cla()
        plt.clf()
        plt.close("all")
        succeedNum += 1
        print("Processed : {0:003d} / {1} : {2} :: {3}".format(idx, imgNumber, failedNum, succeedNum))

    print("Succeed : {0}/{1} ; Failed : {2}/{3}".format(succeedNum, imgNumber, failedNum, imgNumber))
    print("Succeed : {0:.3f} ; Failed : {1:.3f}".format(float(succeedNum)/float(imgNumber), float(failedNum)/float(imgNumber)))
