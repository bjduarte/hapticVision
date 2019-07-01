import test_simple

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

pictureName = 'image1'
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path + '\\assets', pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets', pictureName + "new.jpg"))
pictureName = 'image2'
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path + '\\assets', pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets', pictureName + "new.jpg"))

pictureName = 'image3'
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path + '\\assets', pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets', pictureName + "new.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
print(detections)
