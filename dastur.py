import cv2
import time
import face_recognition
import mediapipe as mp


cap=cv2.VideoCapture("./1.mp4")
ptime=0
mpFaceDetection=mp.solutions.face_detection
mpdraw=mp.solutions.drawing_utils
FaceDetection=mpFaceDetection.FaceDetection()

while True:

    success, img = cap.read()
    imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    natija=FaceDetection.process(imgrgb)
    print(natija)

    if natija.detections:
        for id, detection in enumerate(natija.detections):
            mpdraw.draw_detection(img,detection)
            print(id,detection) # relative boxni ifoda etadi
            print(detection.score)
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            x=int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
              int(bboxC.width*iw), int(bboxC.height*ih)

            cv2.rectangle(img,x,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',
                        (x[0],x[1]-10), cv2.FONT_HERSHEY_PLAIN,
                        1.5,(255,255,0),2)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f"FPS:{int(fps)}",(30,90), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
    cv2.imshow("Dastur", img)
    cv2.waitKey(20)