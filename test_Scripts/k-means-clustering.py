import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt 

# Load the input image and bounding boxes
img = cv2.imread('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\images\\d_r_1_.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change color to RGB (from BGR)
bounding_boxes = [(150, 200, 420, 390), (200, 200, 300, 300)]


# Define the k-means parameters
k = 2
seed= np.asarray([[img.shape[0] // 2, img.shape[1] // 2],[0,0]]) # Use the center of the image as the seed
max_iterations = 100

def k_means(img,bounding_boxes):
# For each bounding box, extract the corresponding image region and apply k-means clustering
    for box in bounding_boxes:
        # Extract the bounding box image region
        x1, y1, x2, y2 = box
        bbox_img = img[y1:y2, x1:x2]
        cv2.imshow('result', bbox_img)
        cv2.waitKey()
        #print(bbox_img)
        # Convert the bounding box image to a suitable format for k-means clustering
        pixels = bbox_img.reshape((-1,3))
        pixels = np.float32(pixels)


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0) #criteria
        k = 5 # Choosing number of cluster
        retval, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS) 
        centers = np.uint8(centers) # convert data into 8-bit values 
        segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
        segmented_image = segmented_data.reshape((bbox_img.shape)) # reshape data into the original image dimensions
        plt.imshow(segmented_image)
        plt.show()
        #pixels = bbox_img
        # Apply k-means clustering
        #kmeans = KMeans(n_clusters=k, init=np.array([bbox_img.mean(axis=(0,1))]*k), max_iter=max_iterations, n_init=1, random_state=start)
        #kmeans = KMeans(n_clusters=k, init=np.array([bbox_img.mean(axis=(0,1))]*k), max_iter=max_iterations)
        '''
        kmeans =KMeans(n_clusters=k, init=seed, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
        kmeans.fit(pixels)
        # Assign cluster labels to pixels
        labels = kmeans.predict(pixels)

        # Reshape the labels array back to the original bounding box image shape
        segmentation = labels.reshape((bbox_img.shape[0], bbox_img.shape[1]))
        print(segmentation)
        # Convert the labels array to an image
        color_map = cv2.COLORMAP_COOL
        label_img = cv2.applyColorMap(np.uint8(segmentation)*255, color_map)
        # Display the segmentation result
        #cv2.imshow('Segmentation result', segmentation.astype(np.uint8)*255)
        cv2.imshow('Segmentation result', label_img)
        cv2.waitKey()
        '''

k_means(img,bounding_boxes)
