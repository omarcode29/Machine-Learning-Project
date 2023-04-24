# Machine-Learning-Project


## Files : 

### Preprocessing Folder -> preprocess.py : preprocessing and saving the images
### Faster_RCNN -> object_detection_transfer_learning.ipynb : object detection model Faster R-CNN with transfer learning
### k-means -> k-mean-mask.py: k-means clustering algorithm on the bounding box images
### YOLOv5 : YOLOv5 model as published by ultalytics: https://github.com/ultralytics/yolov5
### SUIM Folder: containes the SUIM NET as implemented by Islam M. and Edge
C.[1]Islam M., Edge C., Xiao Y., Luo P. : Semantic Segmentation of Underwater Imagery: Dataset and Benchmark. Computer Vision and Pattern Recognition (cs.CV); arXiv:2004.01241 (2020)



The proposed approach involves a semi-supervised learning method for under-
water imagery semantic segmentation. The following steps describe the approch:

## 1. Data Preparation:

The first step in the proposed approach is to download and preprocess the
Segmentation of Underwater IMagery (SUIM) dataset. Specifically, 10% of
the labeled data will be used for supervised learning, while the remaining
data will be utilized as unlabeled data. it is important to set the right paths to the dataset in preprocess.py

## 2. Object Detection:
The next step is to train an object detection model using the labeled data.
To accomplish this, a pre-trained state of the art object detection model
YOLOv5. to set up the yolov5 model: 

-install the requirments (pip install -r requirments.txt). 

-copy the dataset to the datasetsfolder

-run the scripts\make_txt.py to prepare the txt files in the dataset.

-train the model : python train.py --data data/water.yaml --cfg models/yolov5x.yaml --weights weight/yolov5x.pt --batch-size 15 --epochs 80 --workers 4

-to deploy the model: python detect.py (parameters to set the paths) 

## 3. K-Mean Clustering:
In this approach, k-mean clustering will be employed to identify the regions
of the pixels and assign pseudo-labels. The opencv library in Python will be
used to implement k-mean clustering. k is 2, either object or background.

## 4. Encoder-Decoder Architecture:
The encoder-decoder architecture, initially proposed by Islam M. and Edge
C.[1], will be used to perform semantic segmentation. The architecture can
be defined and trained in Tensorflow.
