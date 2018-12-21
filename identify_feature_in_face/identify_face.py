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

def paint_faces(image):
    # 给识别的脸画方框
    faces = get_faces(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

img = cv2.imread(r"./players.png")
processed_img = paint_faces(img)
cv2.imshow("players_with_joker_nose", cv2.resize(processed_img,(800, 500),interpolation=cv2.INTER_CUBIC))

