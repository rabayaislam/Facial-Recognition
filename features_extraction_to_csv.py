import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np


path_images_from_camera = "data/data_faces_from_camera/"


detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_5_face_landmarks.dat")


# Face recognition model, the object maps human faces into 128D vectors
face_rec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")



def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)

    print("%-40s %-20s" % ("image with faces detected:", path_img), '\n')

   
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_rec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("no face")

    return face_descriptor


def return_features_mean_personX(path_faces_personX):
    features_list_personX = []
    photos_list = os.listdir(path_faces_personX)
    if photos_list:
        for i in range(len(photos_list)):
            print("%-40s %-20s" % ("image to read:", path_faces_personX + "/" + photos_list[i]))
            features_128d = return_128d_features(path_faces_personX + "/" + photos_list[i])
        
            if features_128d == 0:
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        print("Warning: No images in range" + path_faces_personX + '/', '\n')
 
    
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX).mean(axis=0)
    else:
        features_mean_personX = '0'

    return features_mean_personX


# get the num of latest person
person_list = os.listdir("data/data_faces_from_camera/")
person_num_list = []
for person in person_list:
    person_num_list.append(int(person.split('_')[-1]))
person_cnt = max(person_num_list)

with open("data/features_all.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for person in range(person_cnt):
        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        print(path_images_from_camera + "person_"+str(person+1))
        features_mean_personX = return_features_mean_personX(path_images_from_camera + "person_"+str(person+1))
        writer.writerow(features_mean_personX)
        print("The mean of features:", list(features_mean_personX))
        print('\n')
    print("Save all the features of faces registered into: data/features_all.csv")
    
    
