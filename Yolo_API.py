import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from model import Yolov1
from utils import non_max_suppression, cellboxes_to_boxes


def plot_image(image, boxes, font_size=6):
    # pascal voc labels as per id
    pascal_voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                         'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Each box is of format : [class_label,object_prob,x,y,w,h]

    objects_detected = []

    # Create a Rectangle potch
    for box in boxes:
        box_label = pascal_voc_labels[int(box[0])]
        objects_detected.append(box_label)
        obj_prob = round((box[1] * 100), 1)
        bbox_label = box_label + " : " + str(obj_prob) + "%"

        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Displays object name on bbox
        rx, ry = rect.get_xy()
        ax.annotate(bbox_label, (rx, ry), color='white', backgroundcolor='lime',
                    weight='bold', fontsize=font_size, ha='left', va='bottom')

    if len(objects_detected) < 1:
        print("NO OBJECTS FROM DETECTION LIST DETECTED.")
    else:
        print("OBJECTS DETECTED IN IMAGE : ", objects_detected)
    plt.show()


# Takes an Image Path as input and plots bounding boxes on
# objects detected on image,then displays the image back.
def detect_objects(image_path, font_size=6, weights_path=None):
    assert type(image_path) == str, "Image path should be a string."

    img = cv2.imread(image_path)
    # Input shape of Yolo V1
    input_shape = (448, 448)
    img = cv2.resize(img, input_shape)

    # reshaping image as needed by pytorch
    frame = np.array(img)
    frame = frame.reshape(-1, 3, input_shape[0], input_shape[1])
    frame = torch.Tensor(frame)

    # ================================================

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)

    # Loading pretrained weights
    if weights_path:
        print("Loading Pre-trained weights...")
        # If Gpu available
        if torch.cuda.is_available():
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

    # Predicting bboxes on the image.
    bboxes = model(frame)

    bboxes = cellboxes_to_boxes(bboxes)
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    plot_image(img, bboxes, font_size)
