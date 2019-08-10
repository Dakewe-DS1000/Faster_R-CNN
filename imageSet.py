import os  
import random  
#Train    : Data set for Training
#TrainVal : Union for Training and Testing
#Test     : Data set for Testing
#Val      : Data set for Validation
trainval_percent = 0.8 #dataset for training :: percent of all dataset default = 0.8
train_percent = 0.9 #dataset for training validation :: percent of trainval default = 0.9
xmlfilepath = "data\\VOCdevkit2007\\VOC2007\\Annotations"  #Annotations
txtsavepath = "data\\VOCdevkit2007\\VOC2007\\ImageSets\\Main"
total_xml = os.listdir(xmlfilepath)
  
num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)
  
ftrainval = open('data\\VOCdevkit2007\\VOC2007\\ImageSets\\Main\\trainval.txt', 'w')
ftest = open('data\\VOCdevkit2007\VOC2007\\ImageSets\\Main\\test.txt', 'w')
ftrain = open('data\\VOCdevkit2007\VOC2007\\ImageSets\\Main\\train.txt', 'w')
fval = open('data\\VOCdevkit2007\\VOC2007\\ImageSets\\Main\\val.txt', 'w')
  
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()  