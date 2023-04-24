
import os
import sys
import errno
#from sklearn.model_selection import train_test_split
import shutil
import random

'''
root = '/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/'
yolo_path = 'yolov5/datasets/RDD2022'
path = '/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/Norway/train'
countries = os.listdir(root)
for country in countries:
    if country not in ['Norway', 'India', 'Czech']:
        continue

    new_path_images = os.path.join(yolo_path,country, 'images')
    new_path_annotations = os.path.join(yolo_path,country, 'annotations')
    try:
        os.makedirs(new_path_images)
        os.makedirs(new_path_annotations)
    except OSError as e:
        print("warning! {} already exists".format(new_path_images))
        if e.errno != errno.EEXIST:
            raise
    images_path = os.path.join(root, country, 'train/images')
    image_list = os.listdir(images_path)
    annotations_path = os.path.join(root, country, 'train/annotations/xmls')
    annotations_list = os.listdir(annotations_path)

    for img in image_list:
        full_image_path = os.path.join(images_path, img)
        cmd = 'cp ' + full_image_path + ' ' + new_path_images
        print(cmd)
        os.system(cmd)

    for annotation in annotations_list:
        full_annotaion_path = os.path.join(annotations_path, annotation)
        cmd = 'cp ' + full_annotaion_path + ' ' + new_path_annotations
        print(cmd)
        os.system(cmd)   

'''


#root = '/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/'
root_train = '/cluster/home/omarabd/ML_project/yolov5/datasets/underwater-images/train/images'
root_val = '/cluster/home/omarabd/ML_project/yolov5/datasets/underwater-images/valid/images'
#yolo_path = 'yolov5/datasets/RDD2022'
#path = '/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/Norway/train'
#images = os.listdir(root)

train_images = [os.path.join(root_train, x) for x in os.listdir(root_train)]
train_images.sort()

val_images = [os.path.join(root_val, x) for x in os.listdir(root_val)]
val_images.sort()
#train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
#random.shuffle(images)
#train_images = images[:int((len(images)+1)*.80)] #Remaining 80% to training set
#val_images = images[int((len(images)+1)*.80):] #Splits 20% data to test set
#val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)


#print(sys.path[0])
#/cluster/home/omarabd/rddc2020-master/yolov5/datasets/road2020/train.txt
file = open('/cluster/home/omarabd/ML_project/yolov5/datasets/underwater-images/train.txt','w')
for item in train_images:
    file.write(item+"\n")
file.close()

file = open('/cluster/home/omarabd/ML_project/yolov5/datasets/underwater-images/val.txt','w')
for item in val_images:
    file.write(item+"\n")
file.close()
