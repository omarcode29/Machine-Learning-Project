import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, segmentation, color

# Load the image
#img = io.imread('underwater_image.png')
img = io.imread('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\images\\d_r_1_.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change color to RGB (from BGR)
bounding_boxes = [(150, 200, 420, 390), (200, 200, 300, 300)]
# Define the seed point as the center of the image
seed_point = (img.shape[0]//2, img.shape[1]//2)
for box in bounding_boxes:
    # Extract the bounding box image region
    x1, y1, x2, y2 = box
    bbox_img = img[y1:y2, x1:x2]
    seed_point = (bbox_img.shape[0]//2, bbox_img.shape[1]//2)
    cv2.imshow('result', color.rgb2gray(bbox_img))
    cv2.waitKey()

    # Perform region growing segmentation
    labels = segmentation.flood(color.rgb2gray(bbox_img), seed_point)

    # Display the segmented image
    plt.imshow(labels)
    plt.show()