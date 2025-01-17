import dlib
import numpy as np  
import cv2          
import os           
import shutil       


# frontal face detector
detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')


camera = cv2.VideoCapture(0)

#The counter for screen shoot
cnt_ss = 0

#  The folder to save face images
current_face_dir = ""

#  The directory to save images of faces
path_photos_from_camera = "data/data_faces_from_camera/"


# 1. Mkdir for saving photos and csv
def pre_work_mkdir():

    #make folders to save faces images and csv
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)


pre_work_mkdir()

# 2. Delete the old data of faces
def pre_work_del_old_face_folders():
    # data_faces_from_camera/person_x/"...
    folders_rd = os.listdir(path_photos_from_camera)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_photos_from_camera+folders_rd[i])

    if os.path.isfile("data/features_all.csv"):
        os.remove("data/features_all.csv")


# If enable this function, it will delete all the old data in dir person_1/,person_2/,/person_3/...
# pre_work_del_old_face_folders()



# 3. Check people order: person_cnt
# If the old folders exists
# person_x+1  Start from person_x+1
if os.listdir("data/data_faces_from_camera/"):
    #  Get the num of latest person
    person_list = os.listdir("data/data_faces_from_camera/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)


# Start from person_1
else:
    person_cnt = 0

# The flag to control if save
save_flag = 1

# The flag to check if press 'n' before 's'
press_n_flag = 0

while camera.isOpened():
    flag, img_rd = camera.read()
    # print(img_rd.shape)
    # It should be 480 height * 640 width in Windows and Ubuntu by default
    # Maybe 1280x720 in macOS

    kk = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    
    #Faces
    faces = detector(img_gray, 0)

    #Font to write
    font = cv2.FONT_ITALIC

    #press 'n' to create the folders for saving faces
    if kk == ord('n') or kk == ord('N'):
        person_cnt += 1
        current_face_dir = path_photos_from_camera + "person_" + str(person_cnt)
        os.makedirs(current_face_dir)
        print('\n')
        print("Create folders: ", current_face_dir)

        cnt_ss = 0              #clear the cnt of faces
        press_n_flag = 1        #have pressed 'n'

    #Face detected
    if len(faces) != 0:
        #Show the rectangle box of face
        for k, d in enumerate(faces):
            # Compute the width and height of the box
            # (x,y), (width, height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # compute the size of rectangle box
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/2)
            ww = int(width/2)

            #the color of rectangle of faces detected
            color_rectangle = (0, 255, 0)

            #480x640
            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(img_rd, "SORRY,OUT OF RANGE", (20, 300), font, 1.1, (0, 0, 255), 2, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
                if kk == ord('s') or kk == ord('S'):
                    print("Please adjust your position")
            else:
                color_rectangle = (0, 255, 0)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)

            #Create blank image according to the shape of face detected
            im_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

            if save_flag:
                #Press 's' to save faces into local images
                if kk == ord('s') or kk == ord('S'):
                    #check if you have pressed 'n'
                    if press_n_flag:
                        cnt_ss += 1
                        for ii in range(height*2):
                            for jj in range(width*2):
                                im_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_ss) + ".jpg", im_blank)
                        print("Save into：", str(current_face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    else:
                        print("Please press 'N' before 'S'")

    #Show the numbers of faces detected
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #Add some statements
    cv2.putText(img_rd, "Please Register your face", (20, 40), font, 1.2, (230, 216, 173), 2, cv2.LINE_AA)
    cv2.putText(img_rd, "N: Create a new face folder", (20, 350), font, 0.9, (232, 185, 124), 2, cv2.LINE_AA)
    cv2.putText(img_rd, "S: Save current face", (20, 400), font, 0.9, (190, 132, 178), 2, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.9, (42, 0, 175), 2, cv2.LINE_AA)

    #Press 'q' to exit
    if kk == ord('q') or kk == ord('Q'):
        break

    #Uncomment this line if you want the camera window is resizeable
    # cv2.namedWindow("camera", 0)

    cv2.imshow("camera", img_rd)

#Release camera and destroy all windows
camera.release()
cv2.destroyAllWindows()