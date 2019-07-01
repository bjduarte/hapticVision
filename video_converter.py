import cv2
import os
import sys
import math
def convertVideoToPictures(video_name):
    if not os.path.exists('assets//videos'):
        os.mkdir('assets//videos')

    vidcap = cv2.VideoCapture('assets//videos//' + video_name)
    success,image = vidcap.read()
    count = 0

    while success:
      cv2.imwrite("assets//videos//frame%d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1
    return(count)

def getPicturesFromVideo(video_name):
    if not os.path.exists('assets//videos'):
        os.mkdir('assets//videos')

    cap = cv2.VideoCapture('assets//videos//' + video_name)
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = 'assets//videos//' +  str(int(x)) + ".jpg"
            x+=1
            cv2.imwrite(filename, frame)

    cap.release()


def removeFrames():
    count = findCount()

    for i in range(1,count):
        if(i % 2 == 0):
            os.remove("assets//videos//frame%d.jpg" % i)

    oddCount = 1
    count = (int)(count / 2)
    for i in range(1,count):

        os.rename("assets//videos//frame%d.jpg" % oddCount, "assets//videos//frame%d.jpg" % i)
        oddCount += 2



def findCount():
    count = 0
    while(os.path.exists("assets//videos//%d.jpg" % (count + 1))):
        count += 1
    return(count)

if __name__ == '__main__':
    #args = parse_args()
    #test_simple(args)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    execution_path = os.getcwd()

    image = cv2.imread(execution_path + '//assets//test_image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Capture frame-by-frame





    # Draw a rectangle around the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    screen_res = 3000, 2000
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt',image)
    cv2.waitKey()
    cv2.destroyAllWindows()
