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
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from imageai.Detection import ObjectDetection
import os

def drawBounds(x1,y1,x2,y2,picture):
    #draws the a black bounding box around a grayscale image
    for i in range(x1,x2 - 1):
        picture[0][0][y1][i] = 0
        picture[0][0][y2-1][i] = 0

    for i in range(y1,y2-1):
        picture[0][0][i][x1] = 0
        picture[0][0][i][x2-1] = 0

def drawBoundsColor(x1,y1,x2,y2,picture):
    #draws black bounding box in a colored pictured
    for o in range(3):
        for i in range(x1,x2):
            picture[y1][i][o] = 0
            picture[y2][i][o] = 0

        for j in range(y1,y2):
            picture[j][x1][o] = 0
            picture[j][x2][o] = 0

def findAverage(x1,y1,x2,y2,picture):
    #finds the average distance for an area in a box for the given processed image through monodepth
    count = 1
    total = 0
    for i in range(x1,x2 - 1):
        for j in range(y1,y2 - 1):
            count += 1
            total += picture[0][0][j][i]
    return(total / count)

def getDistnace(input):
    #covnerts the monodepth value into distance in feet
    return(25.4662 * math.pow(input,(-1.3284)))


def findPixelPerInch(distance):
    #determines the amount of pixels per inch for a given picture. THIS IS DEPENDANT ON THE RESOLUTION OF THE PHOTO, so it will not work for most photos
    if(distance < 80):
        return 43
    return(-0.00000671669535 * distance ** 3 + 0.0050 * distance ** 2 - 1.2050 * distance + 109.3846)

def findAngleRatio(x,y,angle):
    #finds the angle ratio for a given resolution
    return(angle / math.sqrt(x ** 2 + y ** 2))

def findAngle(ratio, x,y,heightC, widthC):
    #finds the angle of a object in a picture
    sign = 1
    if(widthC > x):
        sign = -1
    x = math.sqrt((widthC - x) ** 2)
    y = math.sqrt((heightC - y) ** 2)
    return(sign * ratio * math.sqrt(x ** 2 + y ** 2))

video_name = 'test.mp'

video_converter.convertVideoToPictures(video_name)
amount_pictures = video_converter.findCount()

fps = cv2.VideoCapture('assets//videos//' + video_name).get(5)

execution_path = os.getcwd()
video_path = os.getcwd() + '\\assets\\videos\\'

count = 1

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True,car=True)


height = 0
width = 0
final_images = []

average_time = datetime.now() - datetime.now()

while count < amount_pictures:
    count = count + 1
    current = datetime.now()

    pictureName = ("%d" % count)

    image = cv2.imread(video_path + pictureName + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    test_simple.test_simple_inputs(video_path + pictureName + '.jpg','mono_1024x320', execution_path + '\\assets\\proccessed\\', False)

    average_time = datetime.now() - current + average_time

    current = datetime.now()

    #detections = detector.detectObjectsFromImage(input_image=os.path.join(video_path +  pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets\\proccessed',"new%d.jpg" % count))

    detections = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_image=os.path.join(video_path +  pictureName + ".jpg"),
                                                        output_image_path=os.path.join(execution_path + '\\assets\\proccessed',"new%d.jpg" % count))

    current = datetime.now()
    average_time = datetime.now() - current + average_time
    if(not count == 1):
        average_time /= 2

    print('ETA is ' + str(average_time * amount_pictures))

    numpy_Pic = numpy.load(execution_path + '\\assets\\proccessed\\' + pictureName + '_disp.npy')
    numpy_PicFlat = numpy.ones((numpy_Pic.size,1))

    nPicReal = imread(video_path + pictureName + '.jpg')

    nPicReal = numpy.array(Image.open(video_path + pictureName + '.jpg'))
    nPicReal.setflags(write=1)

    tempcount = 0
    max = 0

    for i in range((len(numpy_Pic[0][0]))):
        for j in range(len(numpy_Pic[0][0][0])):
            numpy_PicFlat[tempcount] = numpy_Pic[0][0][i][j]
            if(max < numpy_PicFlat[count]):
                max = numpy_PicFlat[count]
            tempcount = tempcount + 1


    #finds the resolution of the picture
    constantX = len(numpy_Pic[0][0][0])
    constantY = len(numpy_Pic[0][0])

    #finds the resolution of the picture
    originalX = len(nPicReal[0])
    originalY = len(nPicReal)

    averages = []
    sizeX = []
    sizeY = []

    centerX = []
    centerY = []

    #Used to determine the size of each object
    for eachObject in detections:
        sizeX.append(eachObject['box_points'][2] - eachObject['box_points'][0])
        sizeY.append(eachObject['box_points'][3] - eachObject['box_points'][1])

    #used to determine the center of each object
    for eachObject in detections:
        centerX.append((eachObject['box_points'][2] + eachObject['box_points'][0]) / 2)
        centerY.append((eachObject['box_points'][3] + eachObject['box_points'][1]) / 2)

    #used to scale object sized into the altered monodepth pictures

    tempcount = 0
    #finds the average distance for each box / object that was detected in a picture
    for eachObject in detections:
        x1 = (int)(round(eachObject['box_points'][0] * constantX / originalX))
        x2 = (int)(round(eachObject['box_points'][2] * constantX / originalX))
        y1 = (int)(round(eachObject['box_points'][1] * constantY / originalY))
        y2 = (int)(round(eachObject['box_points'][3] * constantY / originalY))

        averages.append(getDistnace(findAverage(x1,y1,x2,y2,numpy_Pic)))

        drawBounds((x1),(y1),(x2),(y2),numpy_Pic)
        tempcount += 1
    for i in range(len(detections)):
        detections[i].setdefault('average',averages[i])
        detections[i].setdefault('centerX',centerX[i])
        detections[i].setdefault('centerY',centerY[i])

    #finds the angle ratio
    angle = 45
    height, width = (nPicReal.size / (nPicReal[0].size * nPicReal[0][0].size * nPicReal[0][0][0].size)),(nPicReal[0].size /  (  nPicReal[0][0].size * nPicReal[0][0][0].size))
    heightC, widthC = height / 2, width / 2
    ratio = findAngleRatio(width,height,angle)

    # for i in range(len(averages)):
    #     #prints out all the important information regarding the objects in the picture
    #     pixel = findPixelPerInch(averages[i] * 11)
    #     print(detections[i]['name'] + str(averages[i]))
    #     #print('Angle = '+ str(findAngle(ratio,centerX[i],centerY[i],heightC,widthC)))
    mat = numpy_Pic[0][0] / max

    img = Image.fromarray(numpy.uint8(mat * 27755) , 'L')
    img.save("assets//proccessed//Proccessedframe%d.jpg" % count)


    font = ImageFont.truetype('OpenSans-Regular.ttf',24)

    real_picture = Image.fromarray(nPicReal)
    draw = ImageDraw.Draw(real_picture)
    for eachObject in detections:
        if(eachObject['name'] == 'person'):
            draw.rectangle([eachObject['box_points'][0],eachObject['box_points'][1],eachObject['box_points'][2],eachObject['box_points'][3]], fill = None,outline=(0,244,30))
            draw.text((eachObject['centerX'], eachObject['centerY']),str(round(eachObject['average'],2)),(0,255,30),font=font)


    real_picture = numpy.array(real_picture)
    opencv_image = cv2.cvtColor(real_picture, cv2.COLOR_RGB2BGR)

    final_images.append(opencv_image)
    cv2.imwrite("assets//final//Proccessedframe%d.jpg" % count, opencv_image)
    img = Image.fromarray(nPicReal, 'RGB')

width = 1920
height = 1080

#creating video from the three types of images
video_converter.createVideoFromPicture('assets//final//', 'Proccessedframe','distance.mp4',width,height,fps)
video_converter.createVideoFromPicture('assets//processed//', 'disp_','depth.mp4',width,height,fps)
video_converter.createVideoFromPicture('assets//processed//', 'new','objectdetection.mp4',width,height,fps)



#img.show()
#cv2.imwrite("assets//videos//Proccessedframe%d.jpg" % amount_pictures, threshed)
#img.show()
#plt.show()
