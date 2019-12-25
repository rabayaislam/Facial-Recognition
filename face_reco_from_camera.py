import dlib         
import numpy as np   
import cv2           
import pandas as pd  
import os
import winsound
import pyautogui
import smtplib
import config


facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
duration = 1000 #milliseconds
freq = 2000 #Hz
counter = 0

# msg

def send_email(subject, msg):
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(config.EMAIL_ADDRESS, config.PASSWORD)
        message = 'Subject: {}\n\n{}'.format(subject, msg)
        server.sendmail(config.EMAIL_ADDRESS, config.EMAIL_ADDRESS, message)
        server.quit()
        print("Success: Email sent!")
    except:
        print("Email failed to send.")
        
subject = "ALERT!!!"
msg = "An unauthorized person entered your room."

# Compute the e-distance between two 128D features
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 1. Check csv
if os.path.exists("data/features_all.csv"):
    path_features_known_csv = "data/features_all.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)

    
    # The array to save the features of faces in the database
    features_known_arr = []

    
    # Print known faces
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    # Dlib 
    # The detector and predictor will be used
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    # cv2 
    cap = cv2.VideoCapture(0)

    # 3. When the camera is open
    while cap.isOpened():

        flag, img_rd = cap.read()
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
        faces = detector(img_gray, 0)

        #  font to write later
        font = cv2.FONT_ITALIC

        
        # The list to save the positions and names of current faces captured
        pos_namelist = []
        name_namelist = []

        kk = cv2.waitKey(1)

        
        # press 'q' to exit
        if kk == ord('q') or kk == ord('Q'):
            break
        else:
            # when face detected
            if len(faces) != 0:
                # 4. features_cap_arr
                # 4. Get the features captured and save into features_cap_arr
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

                
                # 5. Traversal all the faces in the database
                for k in range(len(faces)):
                    print("##### camera person", k+1, "#####")
                    # Set the default names of faces with "unknown"
                    name_namelist.append("unknown")

                    # the positions of faces captured
                    pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))

                    # For every faces detected, compare the faces in the database
                    e_distance_list = []
                    for i in range(len(features_known_arr)):
                        if str(features_known_arr[i][0]) != '0.0':
                            print("with person", str(i + 1), "the e distance: ", end='')
                            e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            print(e_distance_tmp)
                            e_distance_list.append(e_distance_tmp)
                        else:
                            # person_X
                            e_distance_list.append(999999999)
                    # Find the one with minimum e distance
                    similar_person_num = e_distance_list.index(min(e_distance_list))
                    print("Minimum e distance with person", int(similar_person_num)+1)

                    if min(e_distance_list) < 0.4:
                        ####### person_1, person_2 ... ########
                        # Here you can modify the names shown on the camera
                        name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                        print("May be person "+str(int(similar_person_num)+1))
                        name_namelist[k] = str("Person "+str(int(similar_person_num)+1))\
                        .replace("Person 10","Lorin")\
                        .replace("Person 11","Shanewz")\
                        .replace("Person 12","Shama")\
                        .replace("Person 13","Nusrat")\
                        .replace("Person 14","Toma")\
                        .replace("Person 15","Misty Ma'am")\
                        .replace("Person 16","Shimul")\
                        .replace("Person 17","Pranti")\
                        .replace("Person 18","Shifa")\
                        .replace("Person 1","Dhusor")\
                        .replace("Person 2","Tithy")\
                        .replace("Person 3","Arpita")\
                        .replace("Person 4","Anika")\
                        .replace("Person 5","Souvik")\
                        .replace("Person 6","Adri")\
                        .replace("Person 7","Rabee")\
                        .replace("Person 8","Mithila")\
                        .replace("Person 9","Borna")
                        
                    else:
                        print("Unknown person arrived")
                        winsound.Beep(freq,duration)
                        counter += 1;
                        if counter > 0:
                            myScreenshot = pyautogui.screenshot()
                            myScreenshot.save(r'C:\Users\USER\Desktop\CNN\data\unauthorized\screenshot' + str(counter) + '.jpg')
                            send_email(subject, msg)

                    # draw rectangle
                    for kk, d in enumerate(faces):
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    print('\n')

                # 6. write names under rectangle
                for i in range(len(faces)):
                    cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        print("Faces in camera now:", name_namelist, "\n")

        cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 1, (42, 0, 175), 2, cv2.LINE_AA)
        cv2.putText(img_rd, "Face Recognition Project", (20, 40), font, 1, (232, 185, 124), 2, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("camera", img_rd)

    cap.release()
    cv2.destroyAllWindows()

else:
    print('##### Warning #####', '\n')
    print("'features_all.py' not found!")
    print("Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'", '\n')
    print('##### Warning #####')