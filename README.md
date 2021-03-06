# Object Detection with YOLOV1

#### This is an Implementation of Object Detection using YOLO V1 trained on Pascal VOC Dataset with over 85 Million Parameters.

#### This model can detect only the objects listed in the above detection list.

```python
DETECTION LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle','bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

NOTE : For Real-Time object detection make sure you have a good memory and Gpu on your computer system,a single forward pass takes on average 700-800 MB of memory.On a NVIDIA TITANX GPU this model can detect objects at 40-90 FPS which makes it very suited for Real-Time Object Detction.
                     
                     
![](/imgs/pascalvoc.png)                     


#### => Images are resized to 448x448 since the model accepts a fixed size image as input.

#### => Each bounding box displays the object name it has detected and the probability of an object being present inside the bounding box.

***

### STEPS TO USE THIS MODEL AS API :

Step 1] Install all required libraries and dependencies named below using Pip.

(torch , cv2 , numpy , matplotlib , tqdm , os , pandas , PIL)

Step 2] Download this repo and open a new project with the main file being main.py

Step 3] Download the pretrained weights required for the model from [here](https://www.kaggle.com/deepeshdm/yolo-v1-pretrained-weights?select=YoloV1_Weights_1000examples.pth)

Step 4] The detect_objects( ) function in main.py acts as an interface to the model,pass the location of your image & weights file to the function & it'll plot back a new image with objects detected.

```python

# Insert the location of your Image below
IMAGE_PATH = r'C:\Users\dipesh\Desktop\Pascal Voc\images\000025.jpg'

# Insert the location of weights file below
WEIGHTS_PATH = r'C:\Users\dipesh\Downloads\YoloV1_Weights.pth'

from Yolo_API import detect_objects
detect_objects(image_path=IMAGE_PATH,weights_path=WEIGHTS_PATH)

```

#### NOTE : The model sometimes produces INCORRECT PREDICTIONS & needs to be trained for more epochs to increase accuracy.
#### (GPU RECOMMENDED FOR RUNNING THE MODEL)

***

Original Repository : [here](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO) (Great Job from the Author !)

Changes Made : 

1] Bounding Boxes now Display Class Label & Object Probability.

2] Created a Simple API interface of the model for easy usability by others.

3] Modified some code for faster preprocessing & postprocessing of images.








