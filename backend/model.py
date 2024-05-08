import base64
import copy
from io import BytesIO

import cv2
import keras.api.backend as K
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.api.layers import (
    BatchNormalization,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    concatenate,
)
from keras.api.models import Model
from keras.api.preprocessing import image
from PIL import Image

### GLOBAL VARIABLES
IMAGE_W, IMAGE_H = 448, 448
TRUE_BOX_BUFFER = 50
GRID_H, GRID_W = 14, 14
CLASS = 20
ANCHORS = np.array(
    [
        1.07709888,
        1.78171903,
        2.71054693,
        5.12469308,
        10.47181473,
        10.09646365,
        5.48531347,
        8.11011331,
    ]
)
BOX = int(len(ANCHORS) / 2)
obj_threshold = 0.3
LABELS = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
#### IMAGE READER


class ImageReader(object):
    def __init__(self, IMAGE_H, IMAGE_W, norm=None):
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.norm = norm

    def encode_core(self, image, reorder_rgb=True):
        image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
        if reorder_rgb:
            image = image[:, :, ::-1]
        if self.norm is not None:
            image = self.norm(image)
        return image

    def fit(self, image):
        train_instance = None
        if not isinstance(train_instance, dict):
            train_instance = {"filename": train_instance}

        image_name = train_instance["filename"]
        h, w, c = image.shape
        if image is None:
            print("Cannot find ", image_name)

        image = self.encode_core(image, reorder_rgb=True)

        if "object" in train_instance.keys():

            all_objs = copy.deepcopy(train_instance["object"])

            for obj in all_objs:
                for attr in ["xmin", "xmax"]:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)

                for attr in ["ymin", "ymax"]:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return image
        return image, all_objs


#### MODEL LOADING
def load_model(weight_path):

    def space_to_depth_x2(x):
        return tf.nn.space_to_depth(x, block_size=2)

    input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER, 4))

    # Layer 1
    x = Conv2D(
        32, (3, 3), strides=(1, 1), padding="same", name="conv_1", use_bias=False
    )(input_image)
    x = BatchNormalization(name="norm_1")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(
        64, (3, 3), strides=(1, 1), padding="same", name="conv_2", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_2")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(
        128, (3, 3), strides=(1, 1), padding="same", name="conv_3", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_3")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(
        64, (1, 1), strides=(1, 1), padding="same", name="conv_4", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_4")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(
        128, (3, 3), strides=(1, 1), padding="same", name="conv_5", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_5")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(
        256, (3, 3), strides=(1, 1), padding="same", name="conv_6", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_6")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(
        128, (1, 1), strides=(1, 1), padding="same", name="conv_7", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_7")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(
        256, (3, 3), strides=(1, 1), padding="same", name="conv_8", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_8")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(
        512, (3, 3), strides=(1, 1), padding="same", name="conv_9", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_9")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(
        256, (1, 1), strides=(1, 1), padding="same", name="conv_10", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_10")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(
        512, (3, 3), strides=(1, 1), padding="same", name="conv_11", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_11")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(
        256, (1, 1), strides=(1, 1), padding="same", name="conv_12", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_12")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(
        512, (3, 3), strides=(1, 1), padding="same", name="conv_13", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_13")(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_14", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_14")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(
        512, (1, 1), strides=(1, 1), padding="same", name="conv_15", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_15")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_16", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_16")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(
        512, (1, 1), strides=(1, 1), padding="same", name="conv_17", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_17")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_18", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_18")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_19", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_19")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_20", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_20")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(
        64, (1, 1), strides=(1, 1), padding="same", name="conv_21", use_bias=False
    )(skip_connection)
    skip_connection = BatchNormalization(name="norm_21")(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(
        1024, (3, 3), strides=(1, 1), padding="same", name="conv_22", use_bias=False
    )(x)
    x = BatchNormalization(name="norm_22")(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(
        BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding="same", name="conv_23"
    )(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    model.load_weights(weight_path)

    return model


#### BOUNDING BOX CLASS
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax

        self.confidence = confidence
        self.set_class(classes)

    def set_class(self, classes):
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return self.label

    def get_score(self):
        return self.classes[self.label]


def find_high_class_probability_bbox(netout_scale, obj_threshold):
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]

    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row, col, b, :4]
                    confidence = netout_scale[row, col, b, 4]
                    box = BoundBox(
                        x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes
                    )
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return boxes


#### RESCALE OUTPUT CLASS
class OutputRescaler(object):
    def __init__(self, ANCHORS):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(self, x, axis=-1, t=-100.0):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

    def get_shifting_matrix(self, netout):

        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[..., 0]

        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]

        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:, igrid_w, :] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h, :, :] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_W[:, :, ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_H[:, :, ianchor] = ANCHORSh[ianchor]
        return (mat_GRID_W, mat_GRID_H, mat_ANCHOR_W, mat_ANCHOR_H)

    def fit(self, netout):
        GRID_H, GRID_W, BOX = netout.shape[:3]

        (mat_GRID_W, mat_GRID_H, mat_ANCHOR_W, mat_ANCHOR_H) = self.get_shifting_matrix(
            netout
        )

        netout[..., 0] = (self._sigmoid(netout[..., 0]) + mat_GRID_W) / GRID_W
        netout[..., 1] = (self._sigmoid(netout[..., 1]) + mat_GRID_H) / GRID_H
        netout[..., 2] = (np.exp(netout[..., 2]) * mat_ANCHOR_W) / GRID_W
        netout[..., 3] = (np.exp(netout[..., 3]) * mat_ANCHOR_H) / GRID_H

        netout[..., 4] = self._sigmoid(netout[..., 4])
        expand_conf = np.expand_dims(netout[..., 4], -1)

        netout[..., 5:] = expand_conf * self._softmax(netout[..., 5:])

        return netout


class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):

        self.anchors = [
            BoundBox(0, 0, ANCHORS[2 * i], ANCHORS[2 * i + 1])
            for i in range(int(len(ANCHORS) // 2))
        ]

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap(
            [box1.xmin, box1.xmax], [box2.xmin, box2.xmax]
        )
        intersect_h = self._interval_overlap(
            [box1.ymin, box1.ymax], [box2.ymin, box2.ymax]
        )

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def find(self, center_w, center_h):

        best_anchor = -1
        max_iou = -1

        shifted_box = BoundBox(0, 0, center_w, center_h)

        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou
        return (best_anchor, max_iou)


def nonmax_suppression(boxes, iou_threshold, obj_threshold):
    bestAnchorBoxFinder = BestAnchorBoxFinder([])

    CLASS = len(boxes[0].classes)
    index_boxes = []

    for c in range(CLASS):

        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        sorted_indices = list(reversed(np.argsort(class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    bbox_iou = bestAnchorBoxFinder.bbox_iou(
                        boxes[index_i], boxes[index_j]
                    )
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)

    newboxes = [boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold]

    return newboxes


#### VISUALIZE FUNCTION
def draw_boxes(image, boxes, labels, obj_baseline=0.05, verbose=False):

    def adjust_minmax(c, _max):
        if c < 0:
            c = 0
        if c > _max:
            c = _max
        return c

    image = copy.deepcopy(image)
    image_h, image_w, _ = image.shape
    score_rescaled = np.array([box.get_score() for box in boxes])
    score_rescaled /= obj_baseline

    colors = sns.color_palette("husl", 20)
    for sr, box, color in zip(score_rescaled, boxes, colors):
        xmin = adjust_minmax(int(box.xmin * image_w), image_w)
        ymin = adjust_minmax(int(box.ymin * image_h), image_h)
        xmax = adjust_minmax(int(box.xmax * image_w), image_w)
        ymax = adjust_minmax(int(box.ymax * image_h), image_h)

        text = "{:10} {:4.3f}".format(labels[box.label], box.get_score())
        if verbose:
            print(
                "{} xmin={:4.0f},ymin={:4.0f},xmax={:4.0f},ymax={:4.0f}".format(
                    text, xmin, ymin, xmax, ymax, text
                )
            )
        cv2.rectangle(
            image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=int(sr)
        )
        cv2.putText(
            img=image,
            text=text,
            org=(xmin + 13, ymin + 13),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1e-3 * image_h,
            color=(1, 0, 1),
            thickness=1,
        )

    return image


##### MAIN FUNCTION
model = load_model("./Group10_yoloV2_Voc2012.h5")


def predict(image, shape: tuple[int, int]):
    imageReader = ImageReader(
        IMAGE_H, IMAGE_W=IMAGE_W, norm=lambda image: image / 255.0
    )
    out = imageReader.fit(image)

    X_test = np.expand_dims(out, 0)
    dummy_array = np.zeros((1, 1, 1, 1, TRUE_BOX_BUFFER, 4))
    y_pred = model.predict([X_test, dummy_array])
    netout = y_pred[0]
    outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
    netout_scale = outputRescaler.fit(netout)
    boxes = find_high_class_probability_bbox(netout_scale, obj_threshold)

    iou_threshold = 0.01
    final_boxes = nonmax_suppression(
        boxes, iou_threshold=iou_threshold, obj_threshold=obj_threshold
    )

    ima = draw_boxes(X_test[0], final_boxes, LABELS, verbose=True)
    ima_scaled = (ima * 255).astype("uint8")
    ima_scaled = cv2.resize(ima_scaled, shape)
    return ima_scaled
