import cv2

face_cascade_path = '/usr/local/opt/opencv/share/'\
                    'OpenCV/haarcascades/haarcascade_frontalface_default.xml'
face_cascade  = cv2.CascadeClassifire(face_cascade_path)

ESC_KEY = 27
INTERVAL = 33
cap = cv2.VideoCapture(0)

end_flag, c_frame = cap.read()

while end_flag == True:
    src = c_frame
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)

    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y+h), (255, 0, 0), 2)

    cv2.imshow("src", src)

    end_flag, c_frame = cap.read()

cv2.destroyAllWindows()
cap.release()
