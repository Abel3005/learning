import os
import numpy as np
import torch
from PIL import Image
from numpy import *
import torchvision.transforms as tr
import torchvision
import cv2 as cv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
#import matplotlib.pyplot as plt
from engine import train_one_epoch, evaluate
import torch.nn.functional as F
import torch.optim as optim
import sys
import utils
#import transforms as T

transf = tr.Compose([tr.ToPILImage(),tr.Resize(16), tr.ToTensor(), tr.Normalize(0.5,0.5)])
transf2 = tr.Compose(tr.ToPILImage())
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*1*1,8)
        self.fc2 = nn.Linear(8,4)
        self.fc3 = nn.Linear(4,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16*1*1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
D_PATH = './daegen_net.pth'    
net = Net()
net.load_state_dict(torch.load(D_PATH))


CATEGORY_NAMES = [
        'person', 'background'
        ]

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
def get_prediction(model,img, threshold):
    #img = Image.open(img_path) #Load Image
    #transform = tr.Compose([tr.toTensor()])
    #img = transform(img)
    img = torchvision.transforms.functional.to_tensor(img)
    print(img.size())
    pred = model([img])
    pred_class = [CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_masks = [i for i in list(pred[0]['masks'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    print(pred_t)
    
            
    #pred_boxes = pred_boxes[:int(pred_t+1)]
    #pred_class = pred_class[:int(pred_t+1)]
    #pred_masks = pred_masks[:int(pred_t+1)]
    return pred_boxes, pred_class, pred
    

def main():
    num_classes = 2
    
    PATH = './model.pth'
    num = 194
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    #img_path1 = './data/a0005.jpg'
    #img_path2 = './data/a0006.jpg'
    #img_path3 = './test3.jpg'
    #img_path4 = './data/a0008.jpg'
    #img1 = Image.open(img_path1).convert("RGB")
    #img2 = Image.open(img_path2).convert("RGB")
    #img1.show()ss
    #img2.show()
    capture = cv.VideoCapture(0)
    for j in range(80):
        img_path = './data2/Data{}.jpg'.format(j)
        ret,frame = capture.read()
        #frame = transf2(frame)
        #boxes , pred_cls, pred = get_prediction(model,img_path, 0.5)
        boxes , pred_cls, pred = get_prediction(model,frame, 0.5)
        img_cv = cv.imread(img_path, cv.COLOR_BGR2RGB)
        for i in range(len(pred[0]['masks'])):
        #score = pred[0]['scores'][i]
            mask = pred[0]['masks'][i, 0]
            box = pred[0]['boxes'][i]
            #cv.waitKey()
            #cv.waitKey()
            try:
                print(mask.size())
                mask = mask.mul(255).byte().cpu().numpy()
                mask = mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                mask = cv.resize(mask,dsize=(320,320))
                #np.repeat(mask,3,axis=0)
                
                #print(mask.size)
                
                #mask = np.repeat(mask[np.newaxis,:,:], 3, axis=2)
       
                #mask = np.moveaxis(mask,0,-1)
                #print(mask.size)
                #mask = transf(mask)
                #mask = torch.from_numpy(mask)
                

                #print(mask.size())
                #output = net(mask)
                #print(output)
                cv.imshow('mask',mask)
                #cv.imwrite('./out2/TestMask{}_{}.jpg'.format(j,i),mask)
            except Exception as e:
                print(str(e))
    
        #contours, _ = cv.findContours(
           # mask.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
       # cv.drawContours(img_cv, contours, -1, (255,0,0) , 2, cv.LINE_AA)

    #cv.imshow('img ouput', img_cv)+
    #cv.waitKey()

    
    #sys.stdout = open('mask1.txt','w')
    #print(shape(masks[0]))
    #print(masks[0][0][0][0])
    #plt.imshow(masks[0], cmap = 'gray')
    #x = [torch.rand(3,300,400), torch.rand(3,500,400)]
    #x1 = np.array(img1)
    #x2 = np.array(img2)
    #image_tensor = torchvision.transforms.functional.to_tensor(img1)
    
    #predictions = model([image_tensor])
    
    #predictions.masks[0]
    
if __name__ == "__main__":
    main()
    
    
