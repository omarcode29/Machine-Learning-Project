
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage 
import os

def clust_rgb(image, label, k=5, iters=3):
    # expects img in rgb
    labels = np.array([[0,0,0],[255,255,0],[0,0,255],[255,0,0]])
    img = image.copy()
    h, w, c = img.shape
    orig = image.copy()
    Klusters = np.random.randint(30, 200, size=(k, 3))
    print('init clusters', Klusters)
    for it in range(iters):
        img = image.copy()
        for i in range(h):
            for j in range(w):
                pnt = img[i][j]
                diff = np.sqrt(np.sum((Klusters - pnt) ** 2, axis=1))
                c = np.argmin(diff)
                img[i][j] = Klusters[c]
                #img[i][j] = labels[c]
        loss = 0
        l = []
        for i in range(k):
            Ys, Xs, c = np.where(img == Klusters[i])
            kth_points = orig[Ys, Xs]
            l.append(np.sum(Klusters[i] - kth_points))
            Klusters[i] = np.mean(kth_points, axis=0) # fix: add small positive constant to avoid division by zero
        loss = sum(l)
        print('Cluster centroids at iteration-{}'.format(it + 1), Klusters)
        print('loss at iteration-{}'.format(it + 1), loss)

    #for idx, Kluster in enumerate(Klusters):
    #   img[np.all(img == Kluster, axis=-1)] = labels[idx]

    # Check a circle around the center of the image and assign the label with the most counts
    x_center = int(w/2)
    y_center = int(h/2)
    radius = int(min(w,h)/4)
    mask = (np.arange(w)[None,:] - x_center)**2 + (np.arange(h)[:,None] - y_center)**2 <= radius**2
    circle_labels, circle_counts = np.unique(img[mask], axis=0, return_counts=True)
    if len(circle_labels) > 0:
        most_common_label = circle_labels[np.argmax(circle_counts)]
        print(most_common_label)

        
        img[np.all(img!=most_common_label, axis=-1)] = labels[0]
        img[np.all(img==most_common_label, axis=-1)] = labels[int(label)+1]



    #return img
    # Post-processing step: use morphological operations to group nearby pixels

    # Post-processing step: use morphological operations to group nearby pixels
    # in the same cluster
    #print(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i in range(k):
        mask = cv2.inRange(img, Klusters[i], Klusters[i])

        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        #print(mask)
        # Use contours to remove small regions
        '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                cnt_mask = np.zeros_like(mask)
                cv2.drawContours(cnt_mask, [cnt], 0, 255, cv2.FILLED)
                # Use the new mask to update the image
                #img[cnt_mask != 0] = Klusters[i]
                #cv2.drawContours(mask, [cnt], 0, 0, cv2.FILLED)
        '''
        img[mask != 0] = labels[i]
        #print(img)

    return img

def yolo_to_bbox(yolo_bbox, img_size):
    # yolo_bbox: [x_center, y_center, w, h] normalized by image size
    x_center, y_center, w, h = yolo_bbox
    img_h, img_w ,_ = img_size
    bbox_x = int((x_center - w/2) * img_w)
    bbox_y = int((y_center - h/2) * img_h)
    bbox_w = int(w * img_w)
    bbox_h = int(h * img_h)
    return [bbox_x, bbox_y, bbox_w, bbox_h]


# define function to read image and bounding boxes
def read_image_and_boxes(image_path, boxes_path):
    # read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Failed to read image file:', image_path)
    
    # read bounding boxes
    with open(boxes_path, 'r') as f:
        boxes_lines = f.readlines()
    boxes = []
    for line in boxes_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError('Invalid line in bounding boxes file:', line)
        class_index = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        box = [x_center, y_center, width, height]
        img_size = image.shape
        box =yolo_to_bbox(box,img_size)
        box.append(class_index)
        box = np.array(box)
        boxes.append(box)
    
    return image, boxes


def bbox_to_mask(image, bboxes, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    #print(mask)
    for box in bboxes:
        x_center, y_center, width, height, class_index = box
        print('Bounding box: class={}, x_center={}, y_center={}, width={}, height={}'.format(class_index, x_center, y_center, width, height))
        #bbox_x, bbox_y, bbox_w, bbox_h ,classidx = bbox
        
        img_bbox = image[y_center:y_center+ height, x_center:x_center+width]
        mask_bb = clust_rgb(img_bbox,class_index, k=2,iters = 10)
        #print(mask_bb)
        mask[y_center:y_center+ height, x_center:x_center+width] = mask_bb

        #mask[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = 255
    return mask

def process_image():
    pass



def get_files(path):
    # Define the path where your files are located


    # Create an empty list to store filenames without extensions
    file_names = []

    # Loop through all the files in the specified path
    for file in os.listdir(path):
        # Check if the file is a JPG image
        if file.endswith('.txt'):
            # Remove the extension from the file name
            file_name_without_extension = os.path.splitext(file)[0]
            file_names.append(file_name_without_extension)
    
    return file_names

if __name__ == '__main__':
    #image=cv2.imread('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\masks\\d_r_1_.bmp')
    path = 'C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\yolo_output\\train_val\\'
    files = get_files(path)
    for file in files:

        image_tag = file
        image_path = 'C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\OutputImages\\'+image_tag+'.jpg'
        bbox_path = 'C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\yolo_output\\train_val\\'+image_tag+'.txt'
        image , bboxes = read_image_and_boxes(image_path,bbox_path)
        img_size = image.shape
        # iterate through boxes
        '''
        for box in bboxes:
            x_center, y_center, width, height, class_index = box
            print('Bounding box: class={}, x_center={}, y_center={}, width={}, height={}'.format(class_index, x_center, y_center, width, height))
            bbox_to_mask(image,box,img_size)
        '''
        result = bbox_to_mask(image,bboxes,img_size )
        #print(result)

        #clusters=clust_rgb(image,k=2,iters = 10)
        #print(image)
        #cv2.imshow('original_image',image)
        result_path = ('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\pseudo_labels\\'+image_tag+'.bmp')
        cv2.imwrite(result_path, result)
        #cv2.imshow('clustered_image',result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()