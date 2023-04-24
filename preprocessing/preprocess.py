import cv2
import numpy as np
import os
import natsort
import random

# Load an underwater image to test
img = cv2.imread('C:\\Users\\oabdu\\ML_Project\\train_val\\train_val\\images\\d_r_1_.jpg')
og_img = img

def preprocess(img):
    # Color Correction
    # Use automatic color balance to correct the color of the image
    clahe = cv2.createCLAHE(clipLimit=2 , tileGridSize=(4,4))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    #
    #a = clahe.apply(a)
    #b = clahe.apply(b)

    # Apply contrast stretching to the L channel
    l = clahe.apply(l)

    # Image Enhancement
    # Use contrast stretching to enhance the contrast of the image
    p2, p98 = np.percentile(l, (2, 98))
    l = cv2.normalize(l, None, alpha=p2, beta=p98, norm_type=cv2.NORM_MINMAX)
    l = cv2.convertScaleAbs(l)

    l = cv2.medianBlur(l, 3)

    lab = cv2.merge((l,a,b))

    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_clahe
    #cv2.imshow('Color Corrected Image', img_clahe) 

def random_files(num, list_): 
  file_names = []
  while True: 
    ap = random.choice(list_) 
    if ap not in file_names: 
        file_names.append(ap) 
        if len(file_names) == num: 
            return file_names 
        
        

if __name__ == '__main__':
    
    # folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/HE"
    folder = "C:\\Users\\oabdu\\ML_Project\\TEST\\TEST"
    # folder = "C:/Users/Administrator/Desktop/Databases/Dataset"

    path = folder + "\\images"
    path_masks = folder + "\\masks"

    files = os.listdir(path)
    files =  natsort.natsorted(files)

    files_masks = os.listdir(path_masks)
    files_masks =  natsort.natsorted(files_masks)

    #num_images = 200
    #random_images = random_files(num_images,files)

    for i in range(len(files)):
        file = files[i]
        filepath = path + "/" + file
        
        prefix = file.split('.')[0]
        filepath_mask = path_masks + "/" + prefix+ '.bmp'
        if os.path.isfile(filepath_mask):
            print('********    file mask  ********', prefix+ '.bmp')
            # img = cv2.imread('InputImages/' + file)
            img = cv2.imread(folder + '\\masks\\' + prefix+ '.bmp')

            output_masks =folder +r'\\OutputMasks\\'
            isExist = os.path.exists(output_masks)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(output_masks)
                print("The new directory is created!")
            if not cv2.imwrite(output_masks + prefix + '_CLAHE.bmp', img):
               raise Exception("Could not write image")
            #cv2.imwrite('OutputImages/' + prefix + '_CLAHE.jpg', preprocess_img)
            #cv2.imwrite(folder +'\\Temps\\' + prefix + '_CLAHE.jpg', preprocess_img)

        if os.path.isfile(filepath):
            print('********    file   ********',file)
            # img = cv2.imread('InputImages/' + file)
            img = cv2.imread(folder + '\\images\\' + file)
            preprocess_img = preprocess(img)
            #cv2.imshow('Color Corrected Image', preprocess_img) 
            output_images =folder +r'\\OutputImages\\'
            isExist = os.path.exists(output_images)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(output_images)
                print("The new directory is created!")
            if not cv2.imwrite(output_images + prefix + '_CLAHE.jpg', preprocess_img):
               raise Exception("Could not write image")
            #cv2.imwrite('OutputImages/' + prefix + '_CLAHE.jpg', preprocess_img)
            #cv2.imwrite(folder +'\\Temps\\' + prefix + '_CLAHE.jpg', preprocess_img)


'''
# Image Enhancement
# Use contrast stretching to enhance the contrast of the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
p2, p98 = np.percentile(gray, (2, 98))
img = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
img = cv2.convertScaleAbs(img)
img_enhance = img
cv2.imshow('Contrast Stretched Image', img_enhance)

# Image Filtering
# Use median filtering to remove noise from the image
img = cv2.medianBlur(img_clahe, 3)
img_filter = img
cv2.imshow('Filtered Image', img_filter)



# Image Registration
# Use feature-based registration to align the image
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(template, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
good = matches[:10]
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))


# Data Augmentation
# Use random cropping and flipping to augment the dataset
rows,cols,_ = img.shape
rand_x = np.random.randint(0, rows - 224)
rand_y = np.random.randint(0, cols - 224)
cropped = img[rand_x:rand_x+224, rand_y:rand_y+224]
flipped = cv2.flip(cropped, 1)

'''
'''
# Display the processed images
cv2.imshow('Original Image', og_img)

cv2.imshow('Color Corrected Image', preprocess(og_img))


#cv2.imshow('Registered Image', img)
#cv2.imshow('Augmented Image', flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''