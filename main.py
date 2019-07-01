import numpy
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import test_simple
import cv2
import video_converter
import math
from datetime import datetime

from imageai.Detection import ObjectDetection
import os
#draws the a black bounding box around a grayscale image
def drawBounds(x1,y1,x2,y2,picture):
    for i in range(x1,x2 - 1):
        picture[0][0][y1][i] = 0
        picture[0][0][y2-1][i] = 0

    for i in range(y1,y2-1):
        picture[0][0][i][x1] = 0
        picture[0][0][i][x2-1] = 0

#Draws black bounding box in a colored pictured
def drawBoundsColor(x1,y1,x2,y2,picture):
    for i in range(x1,x2):
        picture[y1][i][0] = 0
        picture[y2][i][0] = 0

    for i in range(y1,y2):
        picture[i][x1][0] = 0
        picture[i][x2][0] = 0

#finds the average distance for an area in a box for a the given processed image through monodepth
def findAverage(x1,y1,x2,y2,picture):
    count = 1
    total = 0
    for i in range(x1,x2 - 1):
        for j in range(y1,y2 - 1):
            count += 1
            total += picture[0][0][j][i]
    return(total / count)
#covnerts the monodepth value into distance in feet
def getDistnace(input):
    return(25.4662 * math.pow(input,(-1.3284)))

#determines the amount of pixels per inch for a given picture. THIS IS DEPENDANT ON THE RESOLUTION OF THE PHOTO, so it will not work for most photos
def findPixelPerInch(distance):
    if(distance < 80):
        return 43
    return(-0.00000671669535 * distance ** 3 + 0.0050 * distance ** 2 - 1.2050 * distance + 109.3846)

#finds the angle ratio for a given resolution
def findAngleRatio(x,y,angle):
    return(angle / math.sqrt(x ** 2 + y ** 2))
#finds the angle of a object in a picture
def findAngle(ratio, x,y,heightC, widthC):
    sign = 1
    if(widthC > x):
        sign = -1
    x = math.sqrt((widthC - x) ** 2)
    y = math.sqrt((heightC - y) ** 2)
    return(sign * ratio * math.sqrt(x ** 2 + y ** 2))


#converts a video into a picture
video_converter.getPicturesFromVideo('test.mp4')
pictureAmount = video_converter.findCount()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#gets the current directory
execution_path = os.getcwd()
path = os.getcwd() + '\\assets\\videos\\'

#start count of photos.
count = 1

#loads ImageAi object detection model
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

#iterates through all the pictures for the given video, every second one frame is output. This runs on each individual frame/picture
while(count < pictureAmount):
    #timer to keep track of how long it takes for the model to load
    current = datetime.now()

    #name of the pictuers, Named 1.jpg,2.jpg, and so on. This while loop iterates through all of them
    pictureName = ("%d" % count)

    #converts the image into grayscale
    image = cv2.imread(path + pictureName + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #runs monodepth on the picture
    test_simple.test_simple_ethan(path + pictureName + '.jpg','mono_1024x320')
    print('Finished Monodepth2 ' + str(datetime.now() - current))
    current = datetime.now()

    #runs object detection on the picture
    detections = detector.detectObjectsFromImage(input_image=os.path.join(path +  pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets\\proccessed', pictureName + "new.jpg"))
    print('Object detecting ' + str(datetime.now() - current))
    current = datetime.now()
    nPic = numpy.load(path + pictureName + '_disp.npy')
    nPicFlat = numpy.ones((nPic.size,1))
    #loads up the monodepth pictures

    nPicReal = imread(path + pictureName + '.jpg')
    tempcount = 0
    max = 0
    #creates a flatten version of the picture
    for i in range((len(nPic[0][0]))):
        for j in range(len(nPic[0][0][0])):
            nPicFlat[tempcount] = nPic[0][0][i][j]
            if(max < nPicFlat[count]):
                max = nPicFlat[count]
            tempcount = tempcount + 1


    #finds the resolution of the picture
    constantX = len(nPic[0][0][0])
    constantY = len(nPic[0][0])

    #finds the resolution of the picture
    originalX = len(nPicReal[0])
    originalY = len(nPicReal)

    averages = []
    sizeX = []
    sizeY = []

    centerX = []
    centerY = []
    for (x,y,w,h) in faces:
        x1 = (int)(round(x * constantX / originalX))
        y1 = (int)(round(y * constantY / originalY))
        x2 = (int)(round(w * constantX / originalX))
        y2 = (int)(round(h * constantY / originalY))
        drawBounds(x1,y1,x2,y2,nPic)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
                x1 = (int)(round(ex * constantX / originalX))
                y1 = (int)(round(ey * constantY / originalY))
                x2 = (int)(round(ew * constantX / originalX))
                y2 = (int)(round(eh * constantY / originalY))
                drawBounds(x1,y1,x2,y2,nPic)

    #Used to determine the size of each object
    for eachObject in detections:
        sizeX.append(eachObject['box_points'][2] - eachObject['box_points'][0])
        sizeY.append(eachObject['box_points'][3] - eachObject['box_points'][1])

    #used to determine the center of each object
    for eachObject in detections:
        centerX.append((eachObject['box_points'][2] + eachObject['box_points'][0]) / 2)
        centerY.append((eachObject['box_points'][3] + eachObject['box_points'][1]) / 2)

    #used to scale object sized into the altered monodepth pictures
    for eachObject in detections:
        x1 = round(eachObject['box_points'][0] * constantX / originalX)
        x2 = round(eachObject['box_points'][2] * constantX / originalX)
        y1 = round(eachObject['box_points'][1] * constantY / originalY)
        y2 = round(eachObject['box_points'][3] * constantY / originalY)
        eachObject['box_points'] = [x1,y1,x2,y2]

    tempcount = 0
    #finds the average distance for each box / object that was detected in a picture
    for eachObject in detections:
        x1 = (int)(eachObject['box_points'][0])
        y1 = (int)(eachObject['box_points'][1])
        x2 = (int)(eachObject['box_points'][2])
        y2 = (int)(eachObject['box_points'][3])
        averages.append(getDistnace(findAverage(x1,y1,x2,y2,nPic)))
        drawBounds((x1),(y1),(x2),(y2),nPic)
        tempcount += 1

    #finds the angle ratio
    angle = 45
    height, width = (nPicReal.size / (nPicReal[0].size * nPicReal[0][0].size * nPicReal[0][0][0].size)),(nPicReal[0].size /  (  nPicReal[0][0].size * nPicReal[0][0][0].size))
    heightC, widthC = height / 2, width / 2
    ratio = findAngleRatio(width,height,angle)

    for i in range(len(averages)):
        #prints out all the important information regarding the objects in the picture
        pixel = findPixelPerInch(averages[i] * 11)
        print(detections[i]['name'] + str(averages[i]) + 'width = ' + str(sizeX[i] / pixel) + 'height = ' + str(sizeY[i] / pixel))
        print('Angle = '+ str(findAngle(ratio,centerX[i],centerY[i],heightC,widthC)))


    mat = nPic[0][0] / max

    img = Image.fromarray(numpy.uint8(mat * 27755) , 'L')
    img.save("assets//proccessed//Proccessedframe%d.jpg" % count)
    count += 1
        #cv2.imwrite("assets//videos//Proccessedframe%d.jpg" % pictureAmount, threshed)
        #img.show()
        #plt.show()
