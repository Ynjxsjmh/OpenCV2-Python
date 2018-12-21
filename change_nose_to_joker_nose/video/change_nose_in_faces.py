import numpy as np
import cv2

def get_faces(image):
    # 创建 classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    face_cascade.load(r"..\..\data\haarcascades\haarcascade_frontalface_alt.xml")

    # 设定灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 识别面部
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=2,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces

def paint_nose_in_face_with_joker_nose(image):
    nose_pic = cv2.imread(r"./joker_nose.jpg")
    mask = 255 * np.ones(nose_pic.shape, nose_pic.dtype)
    
    faces = get_faces(image)

    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    nose_cascade.load("..\..\data\haarcascades\haarcascade_mcs_nose.xml")
    i = 0
    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            face_part_in_image = image[int(y):int(y+h), int(x):int(x+w)]

            # 识别鼻子
            noses = nose_cascade.detectMultiScale(
                face_part_in_image,
                scaleFactor=1.3,
                minNeighbors=2,
                minSize=(10, 10),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 给识别的鼻子画方框
            for (nx, ny, nw, nh) in noses:
                # Warning: you should add precious x and y to nose's x and y,
                # since the nose's x y is in face_part not the whole image
                nose_center = (int(nx+nw/2+x), int(ny+nh/2+y))
                image = cv2.seamlessClone(nose_pic, image, mask, nose_center, cv2.NORMAL_CLONE)

    return image


capture = cv2.VideoCapture(r"./29435344-1-32.flv_20181220_235956.avi")

while(capture.isOpened()):
    ret, frame = capture.read()  # ret 为帧读取成功标识，prev 为读取到的视频帧
    if ret == False:
        break
    cv2.imshow('frame', paint_nose_in_face_with_joker_nose(frame))

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # press Esc to quit. This doesn't work while two above lines is working
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()

