import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage 

from skimage import measure

def clust_gray(image,k=5,iters=3): # expects img in grayscale
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=image.copy()
    h,w=img.shape
    orig=image.copy()
    Klusters=np.random.randint(0,255,size=k)
    print('init clusters', Klusters)
    for it in range(iters):
        img=image.copy()
        for i in range(h):
            for j in range(w):
                pnt=img[i][j]
                diff=np.abs(Klusters-pnt)
                c=np.argmin(diff)
                img[i][j]=Klusters[c]
        loss=0
        l=[]
        for i in range(k):
            Ys,Xs=np.where(img==Klusters[i])
            kth_points=orig[Ys,Xs]
            l.append(np.sum(Klusters[i]-kth_points))
            Klusters[i]=np.mean(kth_points)
        loss=sum(l)    
        print('Cluster centroids at iteration-{}'.format(it+1), Klusters)
        print('loss at iteration-{}'.format(it+1),loss)
    return img



def clust_rgb(image, k=5, iters=3):
    # expects img in rgb
    img = image.copy()
    h, w, c = img.shape
    orig = image.copy()
    Klusters = np.random.randint(0, 255, size=(k, 3))
    print('init clusters', Klusters)
    for it in range(iters):
        img = image.copy()
        for i in range(h):
            for j in range(w):
                pnt = img[i][j]
                diff = np.sqrt(np.sum((Klusters - pnt) ** 2, axis=1))
                c = np.argmin(diff)
                img[i][j] = Klusters[c]
        loss = 0
        l = []
        for i in range(k):
            Ys, Xs, c = np.where(img == Klusters[i])
            kth_points = orig[Ys, Xs]
            l.append(np.sum(Klusters[i] - kth_points))
            Klusters[i] = np.mean(kth_points, axis=0)+ 1e-8 # fix: add small positive constant to avoid division by zero
        loss = sum(l)
        print('Cluster centroids at iteration-{}'.format(it + 1), Klusters)
        print('loss at iteration-{}'.format(it + 1), loss)
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
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                cnt_mask = np.zeros_like(mask)
                cv2.drawContours(cnt_mask, [cnt], 0, 255, cv2.FILLED)
                # Use the new mask to update the image
                #img[cnt_mask != 0] = Klusters[i]
                #cv2.drawContours(mask, [cnt], 0, 0, cv2.FILLED)

        img[mask != 0] = Klusters[i]
        #print(img)

    return img

def yolo_to_bbox(yolo_bbox, img_size):
    # yolo_bbox: [x_center, y_center, w, h] normalized by image size
    x_center, y_center, w, h = yolo_bbox
    img_h, img_w = img_size
    bbox_x = int((x_center - w/2) * img_w)
    bbox_y = int((y_center - h/2) * img_h)
    bbox_w = int(w * img_w)
    bbox_h = int(h * img_h)
    return [bbox_x, bbox_y, bbox_w, bbox_h]

def bbox_to_mask(bbox, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    mask[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = 255
    return mask

def process_image():
    pass




if __name__ == '__main__':
    image=cv2.imread('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\images\\d_r_1_.jpg')
    box = (150, 200, 420, 390)
    x1, y1, x2, y2 = box
    image = image[y1:y2, x1:x2]
    clusters=clust_rgb(image,k=2,iters = 10)
    cv2.imshow('original_image',image)
    cv2.imshow('clustered_image',clusters)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
