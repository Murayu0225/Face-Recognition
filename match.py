import cv2

IMG_SIZE = (200, 200)

face_cascade = cv2.CascadeClassifier("C:/Users/yumur/AppData/Local/Packages/PythonSoftwareFoundation.Python.3"
                                     ".8_qbz5n2kfra8p0/LocalCache/local-packages/Python38/site-packages/cv2/data"
                                     "/haarcascade_frontalface_default.xml")

img1 = cv2.imread("C:/Users/yumur/Desktop/1.jpg")
img2 = cv2.imread("C:/Users/yumur/Desktop/2.jpg")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1_faces = face_cascade.detectMultiScale(img1_gray, minSize=(100, 100))
img2_faces = face_cascade.detectMultiScale(img2_gray, minSize=(100, 100))

img1_face_rect = img1_faces[0]
img2_face_rect = img2_faces[0]
x1, y1, w1, h1 = img1_face_rect[0],img1_face_rect[1],img1_face_rect[2],img1_face_rect[3]
x2, y2, w2, h2 = img2_face_rect[0],img2_face_rect[1],img2_face_rect[2],img2_face_rect[3]

img1_face = img1[y1:y1+h1, x1:x1+w1]
img2_face = img2[y2:y2+h2, x2:x2+w2]
img1_face = cv2.resize(img1_face, IMG_SIZE)
img2_face = cv2.resize(img2_face, IMG_SIZE)

akaze = cv2.AKAZE_create()

(img1_face_kp, img1_face_des) = akaze.detectAndCompute(img1_face, None)
(img2_face_kp, img2_face_des) = akaze.detectAndCompute(img2_face, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.match(img1_face_des, img2_face_des)

dist = [m.distance for m in matches]
if len(dist) != 0:
    ret = sum(dist) / len(dist)
    print("二人の顔の一致度は、" + str(ret) + "です。")