# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.lower()
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = 0 # Default : int(obj.find('difficult').text) #Modified by yuqiyue 20180108
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    #if not os.path.isfile(cachefile):#if not exist then write to cache file
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename)) # Get XML Information
            #if i % 100==0 :
            #    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        #print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
    #else: #if exit then load from cache file
        # load
    #    with open(cachefile, 'rb') as f:
    #        try:
    #            recs = pickle.load(f)
    #        except:
    #            recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # The Reference Value ==> True Value
        bbox = np.array([x['bbox'] for x in R]) # bbox ==> True Bound Box
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool) # There is no "difficult" label
        det = [False] * len(R)          # What is this mean ??
        #npos = npos + sum(~difficult)   # What is this mean ??
        npos += 1
        #class_recs is the true value
        class_recs[imagename] = {'bbox': bbox, #==> True Bound Box
                                 'difficult': difficult, #==> True difficult
                                 'det': det} #==> True det

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines] #所有的检测结果包括 : 图像名称image_name, 置信度score, 检测位置boundbox (x1, y1, x2, y2)
    image_ids = [x[0] for x in splitlines] #所有图像文件名 image_name
    confidence = np.array([float(x[1]) for x in splitlines]) #检测出的所有置信度 score 0.0~1.0
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #所有检测位置 : (x1, y1, x2, y2)
    #一个图像中可能对应好几个转移性淋巴结，但是每个检测出来的转移性淋巴结对应其图像
    #因此，会有重复的图像文件名
    #图像文件名的数量与置信度的数量和检测位置的数量是一致的

    nd = len(image_ids)     #检测出来的全部淋巴结的数量
    tp = np.zeros(nd)       #Ai认为是转移淋巴结，医生也认为是转移淋巴结
    fp = np.zeros(nd)       #Ai认为是转移淋巴结，医生不认为是转移淋巴结

    if BB.shape[0] > 0: #所有检出的转移性淋巴结的个数
        # sort by confidence
        # 所有的数据，根据置信度进行排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]] #通过文件名匹配真值数据集
            bb = BB[d, :].astype(float)  #获得AI的检测位置作为测试值
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)#获得检测位置的真值
            #实际上就是检测位置的真值与测试值进行比对
            #print(len(BBGT))
            #print(BBGT)
            #通过观察，这里是一个检测位置的真值与一个检测位置的测试值进行比对，看是否位置一致

            if BBGT.size > 0: #如果真值中具有转移淋巴结，当然在这个研究中是不存在否的情况的
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                #以上代码是计算真值数据与测试值数据在位置上是否有重叠
                #如果重叠那么，inters就大于0

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.) -
                        inters)
                #print(uni)
                #print(inters)
                #转移性淋巴结的自动检测研究中，只有一个分类，所以这里的uni和inters都是一个数据的数组
                #如果是多个分类，这里的数组中的数据可能会有多个

                overlaps = inters / uni#计算重叠部分占总面积的多少
                ovmax = np.max(overlaps) #计算最大重叠的面积
                jmax = np.argmax(overlaps)#计算最大重叠面积所对应的分类
                #print(ovmax)
                #print(jmax)
                #import pdb
                #pdb.set_trace()

            if ovmax > ovthresh: #如果真值与测试值发生重叠，则说明检测到了对的转移淋巴结
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1. #AI认为是转移性淋巴结，实际上也是转移性淋巴结
                        R['det'][jmax] = 1 #这个测试值被正确地检测了，做上一个标记
                    else:
                        fp[d] = 1. #AI认为这个是淋巴结，但是这个不是淋巴结 False Positive
            else: #false detection negetive bbox
                fp[d] = 1. #AI认为是转移性淋巴结，实际上不是转移性淋巴结
            
    # compute TPR and FPR
    tpr = np.sum(tp) / float(nd)
    fpr = np.sum(fp)
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    #rec = recall
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    #prec = precision
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)   

    #print(len(roc_tp))
    #print(np.sum(roc_tp))
    #print(np.sum(roc_fp))
    #import pdb
    #pdb.set_trace()
    #print("Voc_eval : TPR={0}, FPR={1}".format(tpr, fpr))

    return rec, prec, ap, tpr, fpr


def voc_eval_2(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.0,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    print("Evaluate By Python voc_eval_2")

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    #if not os.path.isfile(cachefile):#if not exist then write to cache file
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename)) # Get XML Information
            #if i % 100==0 :
            #    print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        #print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
        pickle.dump(recs, f)
    #else: #if exit then load from cache file
        # load
    #    with open(cachefile, 'rb') as f:
    #        try:
    #            recs = pickle.load(f)
    #        except:
    #            recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname] # The Reference Value ==> True Value
        bbox = np.array([x['bbox'] for x in R]) # bbox ==> True Bound Box
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool) # There is no "difficult" label
        det = [False] * len(R)          # What is this mean ??
        #npos = npos + sum(~difficult)   # What is this mean ??
        npos += 1
        #class_recs is the true value
        class_recs[imagename] = {'bbox': bbox, #==> True Bound Box
                                 'difficult': difficult, #==> True difficult
                                 'det': det} #==> True det

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines] #所有的检测结果包括 : 图像名称image_name, 置信度score, 检测位置boundbox (x1, y1, x2, y2)
    image_ids = [x[0] for x in splitlines] #所有图像文件名 image_name
    confidence = np.array([float(x[1]) for x in splitlines]) #检测出的所有置信度 score 0.0~1.0
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #所有检测位置 : (x1, y1, x2, y2)
    #一个图像中可能对应好几个转移性淋巴结，但是每个检测出来的转移性淋巴结对应其图像
    #因此，会有重复的图像文件名
    #图像文件名的数量与置信度的数量和检测位置的数量是一致的

    nd = len(image_ids)         #检测出来的全部淋巴结的数量

    if BB.shape[0] > 0: #所有检出的转移性淋巴结的个数
        # sort by confidence
        # 所有的数据，根据置信度进行排序
        sorted_ind = np.argsort(-confidence)#根据置信度对序号进行从大到小的排序，并存放到数组sorted_ind
        sorted_scores = np.sort(-confidence)#将排序的置信度放到数组sorted_scores当中，注意，score被设置为负数了
        BB = BB[sorted_ind, :]#根据置信度对BoundBox进行排序，并存放到数组BB中
        image_ids = [image_ids[x] for x in sorted_ind]

        score = np.zeros(nd)        #记录下相对应的置信度阈值
        tpr = np.zeros(nd)          #True Positive Rate
        fpr = np.zeros(nd)          #False Positive Rate
        precision = np.zeros(nd)    #Precision in RP Curve
        accuracy = np.zeros(nd)     #Accuracy of machine learning
        cnt_TP = 0.0                #Ai认为是转移淋巴结，医生也认为是转移淋巴结
        cnt_FP = 0.0                #Ai认为是转移淋巴结，医生认为是非转移淋巴结
        cnt_TN = 0.0                #Ai认为是非转移淋巴结，医生也认为是非转移淋巴结
        cnt_FN = 0.0                #Ai认为是非转移淋巴结，医生认为是转移淋巴结

        # go down dets and mark TPs and FPs
        for d in range(nd):#d :: from 0 to nd
            # 因为置信度被从大打小排序了，所以
            # 置信度排序小于d；Ai判断为转移性淋巴结的结果    ==>正样本
            # 置信度排序大于d；Ai判断为肺转移性淋巴结的结果  ==>负样本
            # 在正样本中对比，若有BB与BBGT的重叠，则为TP
            # 在正样本中对比，若无BB与BBGT的重叠，则为FP
            #Ai判断为转移性淋巴结
            for p in range(d) :
                R_p = class_recs[image_ids[p]] #通过文件名匹配真值数据集
                bb_p = BB[p, :].astype(float)  #获得AI的检测位置作为测试值
                ovmax_p = -np.inf
                BBGT_p = R_p['bbox'].astype(float)#获得检测位置的真值
                if BBGT_p.size > 0:
                # compute overlaps
                # intersection
                    ixmin = np.maximum(BBGT_p[:, 0], bb_p[0])
                    iymin = np.maximum(BBGT_p[:, 1], bb_p[1])
                    ixmax = np.minimum(BBGT_p[:, 2], bb_p[2])
                    iymax = np.minimum(BBGT_p[:, 3], bb_p[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                #以上代码是计算真值数据与测试值数据在位置上是否有重叠
                #如果重叠那么，inters就大于0
                    uni = ((bb_p[2] - bb_p[0] + 1.) * (bb_p[3] - bb_p[1] + 1.) +
                            (BBGT_p[:, 2] - BBGT_p[:, 0] + 1.) * (BBGT_p[:, 3] - BBGT_p[:, 1] + 1.) -
                            inters)
                #转移性淋巴结的自动检测研究中，只有一个分类，所以这里的uni和inters都是一个数据的数组
                #如果是多个分类，这里的数组中的数据可能会有多个
                    overlaps = inters / uni#计算重叠部分占总面积的多少
                    ovmax_p = np.max(overlaps) #计算最大重叠的面积
                    jmax = np.argmax(overlaps)#计算最大重叠面积所对应的分类

                if ovmax_p > ovthresh: #如果真值与测试值发生重叠，则说明检测到了对的转移淋巴结
                    cnt_TP += 1.0       #Ai认为是转移性淋巴结，医生也认为是转移性淋巴结
                else:
                    cnt_FP += 1.0       #AI认为是转移性淋巴结，医生认为不是转移淋巴结
###########################################################################################
            # 在负样本中对比，若有BB与BBGT的重叠，则为TN
            # 在负样本中对比，若无BB与BBGT的重叠，则为FN
            #Ai判断为非转移性淋巴结
            for n in range(d, nd) :
                R_n = class_recs[image_ids[n]] #通过文件名匹配真值数据集
                bb_n = BB[n, :].astype(float)  #获得AI的检测位置作为测试值
                ovmax_n = -np.inf
                BBGT_n = R_n['bbox'].astype(float)#获得检测位置的真值
                if BBGT_n.size > 0: #如果真值中具有转移淋巴结，当然在这个研究中是不存在否的情况的
                # compute overlaps
                # intersection
                    ixmin = np.maximum(BBGT_n[:, 0], bb_n[0])
                    iymin = np.maximum(BBGT_n[:, 1], bb_n[1])
                    ixmax = np.minimum(BBGT_n[:, 2], bb_n[2])
                    iymax = np.minimum(BBGT_n[:, 3], bb_n[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                #以上代码是计算真值数据与测试值数据在位置上是否有重叠
                #如果重叠那么，inters就大于0

                # union
                    uni_n = ((bb_n[2] - bb_n[0] + 1.) * (bb_n[3] - bb_n[1] + 1.) +
                            (BBGT_n[:, 2] - BBGT_n[:, 0] + 1.) * (BBGT_n[:, 3] - BBGT_n[:, 1] + 1.) -
                            inters)
                #print(uni)
                #print(inters)
                #转移性淋巴结的自动检测研究中，只有一个分类，所以这里的uni和inters都是一个数据的数组
                #如果是多个分类，这里的数组中的数据可能会有多个

                    overlaps = inters / uni_n   #计算重叠部分占总面积的多少
                    ovmax_n = np.max(overlaps)  #计算最大重叠的面积
                    jmax = np.argmax(overlaps)  #计算最大重叠面积所对应的分类

                if ovmax_n > ovthresh: #如果真值与测试值发生重叠
                    cnt_TN += 1.0       #Ai认为是非转移性淋巴结，医生认为是转移性淋巴结
                else:
                    cnt_FN += 1.0       #AI认为是非转移性淋巴结，医生认为是非转移性淋巴结           

            score[d] = (float)(sorted_scores[d]) * -1.0
            if (cnt_TP + cnt_FN)==0 :     tpr[d] = 0.0
            else:                       tpr[d] = cnt_TP / (cnt_TP + cnt_FN)

            if(cnt_FP + cnt_TN)==0 :    fpr[d] = 0.0
            else:                       fpr[d] = cnt_FP / (cnt_FP + cnt_TN)

            if(cnt_TP + cnt_FP)==0 :      precision[d] = 0.0
            else:                       precision[d] = cnt_TP / (cnt_TP + cnt_FP)

            if(cnt_TP + cnt_FN + cnt_FP + cnt_TN)==0 : accuracy[d] = 0.0
            else:                       accuracy[d] = (cnt_TP + cnt_TN) / (cnt_TP + cnt_FN + cnt_FP + cnt_TN)

            if d % 100 == 0 :
                print("tp = {0:3f}".format(cnt_TP))
                print("fp = {0:3f}".format(cnt_FP))
                print("tn = {0:3f}".format(cnt_TN))
                print("fn = {0:3f}".format(cnt_FN))
                print("d = {0:3f} ; score：{1:3f} ; tpr: {2:3f} ; fpr: {3:3f} ; precision: {4:3f} ; accuracy: {5:3f}".format(d, score[d], tpr[d], fpr[d], precision[d], accuracy[d]))

            cnt_TP = 0.0
            cnt_FP = 0.0
            cnt_TN = 0.0
            cnt_FN = 0.0

    return score, accuracy, precision, tpr, fpr