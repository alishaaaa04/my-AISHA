import cv2 as cv


cap = cv.VideoCapture(0)

if not (cap.isOpened()):
    print("could not access webcam")
else :
    print("webcam access successful")
def vid_cam():#for a contineous live stream
    ret =True
    while ret: 
        ret,frame=cap.read()
        cv.imshow("Captured Frame", frame)
        cv.waitKey(1)

def img_cam():#for single frames one by one until you quit
    ret=True
    while ret: 
        ret,frame=cap.read()
        return frame
def op_image():
    ret,frame=cap.read()
    cv.imwrite("./AI-SHA/Picture.png",frame)


vid_cam()
