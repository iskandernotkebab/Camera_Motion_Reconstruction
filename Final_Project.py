#!/usr/bin/env python
# coding: utf-8

# ### **CMSC426 Final Project**
# Iskander Lou<br>

# In[251]:


import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates as interp2


# In[252]:


# Helpers Functions

def ReadCameraModel(models_dir):
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"
    intrinsics = np.loadtxt(intrinsics_path)
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    G_camera_image = intrinsics[1:5,0:4]
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size//2])
    LUT = lut.transpose()
    return fx, fy, cx, cy, G_camera_image, LUT

def UndistortImage(image,LUT):
    reshaped_lut = LUT[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)
    return undistorted.astype(image.dtype)

def descriptorMatching(descriptors1, descriptors2, threshold_nn, threshold_dr):
    distances = np.linalg.norm(descriptors1[:, np.newaxis] - descriptors2, axis=2)
    neighborm = np.argwhere(distances < threshold_nn)
    ratios = distances[:, np.argsort(distances, axis=1)[:, 1]] / distances[:, np.argsort(distances, axis=1)[:, 0]]
    doublem = np.argwhere(ratios < threshold_dr)
    return neighborm, doublem


# ### **3.1 Compute Intrinsic Matrix**

# In[253]:


fx, fy, cx, cy, _, LUT = ReadCameraModel('/Users/ken/Desktop/Misc/Data Science/CMSC426/Oxford_dataset_reduced/model')

K = np.array([  [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])
print(K)


# ### **3.2 Load and Demosaic Images**

# In[254]:


images_path = "./Oxford_dataset_reduced/images/"
undistorted_path = "./Results/undistorted_images/"
images = [f for f in os.listdir(images_path)]

for image in images:
    original_path = os.path.join(images_path, image)
    img = cv2.imread(original_path, flags=-1)
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    output_path = os.path.join(undistorted_path, image)
    cv2.imwrite(output_path, color_image)


# ### **3.3 Keypoint Correspondences (For One Pair of Images At First)**

# In[255]:


undistorted_images = sorted(glob.glob(undistorted_path + '/*'))
print(undistorted_images[0])
print(len(undistorted_images))


# In[256]:


sift = cv2.SIFT_create()
flann = cv2.FlannBasedMatcher()
image1 = cv2.imread(undistorted_images[0], flags=-1)
image2 = cv2.imread(undistorted_images[1], flags=-1)


# In[257]:


keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
keypoints1 = [kp for kp in keypoints1]
keypoints2 = [kp for kp in keypoints2]

print("Image 1 type:", type(image1))
print("Image 2 type:", type(image2))
print("Keypoints 1 type:", type(keypoints1))
print("Keypoints 2 type:", type(keypoints2))


# In[258]:


matches = flann.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

print("Matches type:", type(good_matches))
print("Lenght of matches:", len(good_matches))


# In[259]:


output_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None)

plt.imshow(output_img)
plt.axis('off')
plt.show()


# ### **3.4 Estimate Fundamental Matrix (For One Pair of Images)**

# In[260]:


pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)

print(type(pts1))
print(type(pts2))
print(pts1.shape)
print(pts2.shape)
print("Fundamental Matrix:")
print(fundamental_matrix)


# ### **3.5 Recover Essential Matrix (For One Pair of Images)**

# In[261]:


essential_matrix = np.dot(np.dot(K.T, fundamental_matrix), K)

print("Essential Matrix:")
print(essential_matrix)


# ### **3.6 Reconstruct Rotation and Translation Parameters (For One Pair of Images)**

# In[262]:


print(K)


# In[263]:


retval, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, K)

if np.linalg.det(R) < 0:
    R = -R
    t = -t
    
print("Translation:")
print(t)
print("Rotation:")
print(R)


# ### **Now Doing It On a Set of Images**

# In[264]:


len(undistorted_images)


# In[265]:


for i in range(1, min(11, len(undistorted_images)-1)):
    image1 = cv2.imread(undistorted_images[i],flags=-1)
    image2 = cv2.imread(undistorted_images[i+1],flags=-1)
    print(i)


# In[266]:


translations = []
rotations = []

for i in range(1, len(undistorted_images)-1):

    image1 = cv2.imread(undistorted_images[i],flags=-1)
    image2 = cv2.imread(undistorted_images[i+1],flags=-1)

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    fundamental_matrix, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
    essential_matrix = np.dot(np.dot(K.T, fundamental_matrix), K)
    retval, R, t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, K)

    translations.append(t)
    rotations.append(R)
    print(i)


# In[267]:


print(type(translations[0]))
print(type(rotations[0]))
print(len(translations))
print(len(rotations))


# In[268]:


print(translations[0])


# In[269]:


print(rotations[0])


# ### **4. Reconstruct the Trajectory**

# In[270]:


R = rotations.copy()
t = translations.copy()


# In[271]:


camera_centers = [np.zeros((3, 1))]

for i in range(len(R)):
    center = -R[i].T @ t[i]
    camera_centers.append(R[i] @ camera_centers[i] + center)

camera_centers = np.array(camera_centers)

x_coords = camera_centers[:, 0]
y_coords = camera_centers[:, 2]
n = 1

plt.plot(x_coords, y_coords, marker='o')
plt.scatter(x_coords[n], y_coords[n], c='red', s=200)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Camera Path')
plt.show()