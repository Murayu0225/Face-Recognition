import cv2

face_cascade = cv2.CascadeClassifier("C:/Users/yumur/AppData/Local/Packages/PythonSoftwareFoundation.Python.3"
                                     ".8_qbz5n2kfra8p0/LocalCache/local-packages/Python38/site-packages/cv2/data"
                                     "/haarcascade_frontalface_default.xml")

ESC_KEY = 27
INTERVAL = 33

cap = cv2.VideoCapture(0)

end_flag, c_frame = cap.read()

while end_flag == True:
    src = c_frame
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)

    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Test", src)

    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    end_flag, c_frame = cap.read()

cv2.destroyAllWindows()
cap.release()
