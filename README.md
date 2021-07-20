# Object Detection with YOLOV1

#### This is an Implementation of Object Detection using YOLO V1 trained on Pascal VOC Dataset with over 85 Million Parameters.

#### This model can detect only the objects listed in the above detection list.

#### DETECTION LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
                     
                     
![](/imgs/pascalvoc.png)                     


#### => Images are resized to 448x448 since the model accepts a fixed size image as input.

#### => Each bounding box displays the object name it has detected and the probability of an object being present inside the bounding box.

#----------------------------------------------------------------------------------

### STEPS TO USE THIS MODEL AS API :

Step 1] Install all required libraries and dependencies named below using Pip.

(torch , cv2 , numpy , matplotlib , tqdm , os , pandas , PIL)

Step 2] Download this repo and open a new project with the main file being main.py

Step 3] Download the training weights required for the model

Step 4] The detect_objects( ) function in main.py acts as an interface to the model,pass the location of your image & weights file to the function & it'll plot back a new image with objects detected.

![](/imgs/yoloapi.png)

#### NOTE : The model sometimes produces INCORRECT PREDICTIONS & needs to be trained for more epochs to increase accuracy.
#### (GPU RECOMMENDED FOR RUNNING THE MODEL)

Original Repository : [here](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO)








