import cv2
import time
import mediapipe as mp


class Yuzni_aniqlash():
    def __init__(self,mindeteccon=0.5):
        self.mindeteccon=mindeteccon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        self.FaceDetection = self.mpFaceDetection.FaceDetection(mindeteccon)

    def yuzni_topish(self,img,draw=True):
        imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.natija=self.FaceDetection.process(imgrgb)
        # print(self.natija)
        D=[] # chegara_qiymatlarini listga joylash
        if self.natija.detections:
            for id, detection in enumerate(self.natija.detections):
                bboxC=detection.location_data.relative_bounding_box
                ih,iw,ic=img.shape
                x=int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                  int(bboxC.width*iw), int(bboxC.height*ih)
                D.append([id,x,detection.score])
                cv2.rectangle(img,x,(255,255,0),2)
                cv2.putText(img,f'{int(detection.score[0]*100)}%',
                            (x[0],x[1]-10), cv2.FONT_HERSHEY_PLAIN,
                            1.5,(0,255,0),2)
        return img, D

def main():
    cap = cv2.VideoCapture("./1.mp4")
    ptime = 0
    detector = Yuzni_aniqlash()
    while True:
        success, img = cap.read()
        img, D=detector.yuzni_topish(img)
        print(D)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f"FPS:{int(fps)}", (5, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        cv2.imshow("Dastur", img)


        if cv2.waitKey(20) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()