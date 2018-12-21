import cv2

def get_faces(image):
    # 创建 classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    face_cascade.load(r"..\data\haarcascades\haarcascade_frontalface_alt.xml")

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
    faces = get_faces(image)

    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    nose_cascade.load("..\data\haarcascades\haarcascade_mcs_nose.xml")
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
                cv2.rectangle(image, (nx+x, ny+y), (nx+nw+x, ny+nh+y), (0, 255, 0), 2)

    return image

img = cv2.imread(r"./players.png")
processed_img = paint_nose_in_face_with_joker_nose(img)
cv2.imshow("after_clever_identification", cv2.resize(processed_img,(800, 500),interpolation=cv2.INTER_CUBIC))

