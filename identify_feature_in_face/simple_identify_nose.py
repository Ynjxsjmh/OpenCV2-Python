import cv2

def get_noses(image):
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    nose_cascade.load("..\data\haarcascades\haarcascade_mcs_nose.xml")

    # 设定灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 识别眼部
    noses = nose_cascade.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=2,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return noses

def paint_noses(image):
    noses = get_noses(image)

    # 给识别出的鼻子画方框
    for (x, y, w, h) in noses:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

img = cv2.imread(r"./players.png")
processed_img = paint_noses(img)
cv2.imshow("identify_noses", cv2.resize(processed_img,(800, 500),interpolation=cv2.INTER_CUBIC))
