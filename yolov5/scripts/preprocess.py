import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd



def crop_and_check_labels(img_path, label_path,target_path):
    # Load image
    img = Image.open(img_path)

    # Load labels in YOLO format
    df = pd.read_csv(label_path, header=None, names=['class', 'x', 'y', 'w', 'h'], delim_whitespace=True)
    boxes = df[['x', 'y', 'w', 'h','class']].values
    #labels = df['class'].values

    # Crop image
    width, height = img.size

    crop_width = width*0.7
    crop_height = height/2
    
    left, top = 0, crop_height
    right, bottom =crop_width, height
    img_crop = img.crop((left, top, right, bottom))

    # Calculate new bounding box coordinates relative to the cropped image
    boxes_crop = boxes.copy()
    boxes_crop[:, 0] = np.clip((boxes_crop[:, 0] * width) - left, 0, crop_width) / crop_width
    boxes_crop[:, 1] = np.clip((boxes_crop[:, 1] * height) - top, 0, crop_height) / crop_height
    boxes_crop[:, 2] = (boxes_crop[:, 2] * width) / crop_width
    boxes_crop[:, 3] = (boxes_crop[:, 3] * height) / crop_height

    # Check if labels are inside the cropped region
    boxes_crop = boxes_crop[np.logical_and(boxes_crop[:, 0] > 0,  boxes_crop[:, 0]< 1)]
    boxes_crop = boxes_crop[np.logical_and(boxes_crop[:, 1] > 0,  boxes_crop[:, 1]< 1)]
    boxes_crop = boxes_crop[np.logical_and(boxes_crop[:, 1] + boxes_crop[:, 3] < 1,  boxes_crop[:, 0] + boxes_crop[:, 2]< 1)]



    #boxes_crop = np.delete(boxes_crop, np.where(np.bitwise_and((1 > boxes_crop[:, 0] > 0) & (1 > boxes_crop[:, 1] > 0) & (boxes_crop[:, 0] + boxes_crop[:, 2]  < 1) & (boxes_crop[:, 1] + boxes_crop[:, 3] < 1)))[0], axis=0)



    # Save cropped image and labels in YOLO format
    head_tail = os.path.split(img_path)
    img_target = os.path.join(target_path, 'images',head_tail[-1])
    txt_target = (img_target[:-4]+".txt").replace('images','labels_cropped')


    img_crop.save(os.path.splitext(img_target)[0] + '_crop.jpg')
    df_crop = pd.DataFrame({'class': (boxes_crop[:, 4]).astype(int), 'x': boxes_crop[:, 0], 'y': boxes_crop[:, 1], 'w': boxes_crop[:, 2] , 'h': boxes_crop[:, 3] })
    df_crop.to_csv(os.path.splitext(txt_target)[0] + '_crop.txt', header=None, index=None, sep=' ', float_format='%.6f')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='directory containing images and YOLO-format labels')
    parser.add_argument('--labels_dir', type=str, required=True, help='directory containing images and YOLO-format labels')
    #parser.add_argument('--crop_size', type=int, default=224, help='size of the cropped region')
    args = parser.parse_args()

    # Iterate over images and labels and crop them
    for img_file in os.listdir(args.data_dir):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(args.data_dir, img_file)
            label_path = os.path.join(args.labels_dir, os.path.splitext(img_file)[0] + '.txt')
            target_path = '/cluster/home/omarabd/rddc2020-master/yolov5/datasets/road2020/'
            crop_and_check_labels(img_path, label_path,  target_path)