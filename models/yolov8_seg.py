import math
import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from utils import random_colors, sigmoid

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# Code and idea originally from ibaiGorordo's architecture.
# https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation
class YOLOv8Seg:
    def __init__(self, conf: float = 0.5, iou: float = 0.5):
        self.num_masks = 32
        self.min_conf = conf
        self.min_iou = iou

        # Model
        self.model_name = 'yolov8m-seg'
        self.model = None

        # Model input dimensions
        self.input_names = None
        self.input_width = None
        self.input_height = None
        self.mask_width = None
        self.mask_height = None
        self.image_width = None
        self.image_height = None

        # Predict results
        self.predict_image = None
        self._result_boxes = None
        self._result_classes = None
        self._result_masks = None
        self._result_confs = None

    @property
    def result_boxes(self):
        return self._result_boxes

    @property
    def result_classes(self):
        return self._result_classes

    @property
    def result_masks(self):
        return self._result_masks

    @property
    def result_confs(self):
        return self._result_confs

    def load_model(self, gpu: bool = False):
        if self.model is None:
            self.__create_model(self.model_name)

            if gpu:
                available_providers = ort.get_available_providers()
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                selected_providers = [p for p in providers if p in available_providers]
                self.model = ort.InferenceSession(f'./weights/{self.model_name}.onnx', providers=selected_providers)
            else:
                self.model = ort.InferenceSession(f'./weights/{self.model_name}.onnx')

            inputs = self.model.get_inputs()
            self.input_names = [inputs[i].name for i in range(len(inputs))]
            self.input_width = inputs[0].shape[3]
            self.input_height = inputs[0].shape[2]

    def __create_model(self, model_name):
        filepath = f'./weights/{model_name}.onnx'

        if os.path.isfile(filepath):
            print(f'{filepath} has already existed.')
        else:
            print(f'Export model {filepath}')
            model = YOLO(f'{model_name}.pt')
            model.export(format="onnx")

            shutil.move(f'./{model_name}.pt', f'./weights/{model_name}.pt')
            shutil.move(f'./{model_name}.onnx', f'./weights/{model_name}.onnx')

    def predict(self, image: np.ndarray):
        # Adjust the input image to match the required input dimensions of the model
        self.predict_image = image
        self.image_height, self.image_width = image.shape[:2]
        input_image = self.preprocess_image(image)

        # Get the predicted results
        outputs = self.model.run(None, {self.input_names[0]: input_image})
        boxes, classes, masks, confs = self.parse_output(outputs)

        # Apply Non-Maximum Suppression to get rid of the bad results
        indices = self.process_NMS(boxes, confs)
        self._result_boxes = boxes[indices]
        self._result_classes = classes[indices]
        self._result_masks = masks[indices]
        self._result_confs = confs[indices]

    def preprocess_image(self, image: np.ndarray):
        # Convert image from BGR channels to RGB channels
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the model input dimensions
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))

        # Scale the values to [0, 1]
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = input_image[np.newaxis, :, :, :]

        return input_image

    def parse_output(self, outputs: np.ndarray):
        # Get the model predictions and masks
        output_predictions = np.squeeze(outputs[0]).T
        output_masks = np.squeeze(outputs[1])

        # Split the predictions
        pred_boxes = output_predictions[:, :4]  # Bounding box (x_center, y_center, width, height)
        pred_classes = output_predictions[:, 4:-self.num_masks]  # Object class detection
        pred_masks = output_predictions[:, -self.num_masks:]  # Segmentation masks

        # Compute the segmentation masks for all detected boxes
        _, self.mask_height, self.mask_width = output_masks.shape
        pred_masks = np.dot(pred_masks, output_masks.reshape(self.num_masks, -1))

        # Remove those useless detected objects not reached the minimum confidence
        confidences = np.max(pred_classes, axis=1)
        pred_boxes = pred_boxes[confidences > self.min_conf]
        pred_classes = np.argmax(pred_classes[confidences > self.min_conf], axis=1)
        pred_masks = pred_masks[confidences > self.min_conf]
        confidences = confidences[confidences > self.min_conf]

        # Convert the bounding boxes and segmentation masks to original image dimensions
        pred_boxes = self.parse_boxes(pred_boxes)
        pred_masks = self.parse_masks(pred_masks, pred_boxes)

        return pred_boxes, pred_classes, pred_masks, confidences

    def parse_boxes(self, boxes):
        # Scale the bounding box to the original image dimensions
        input_shape = (self.input_height, self.input_width)
        output_shape = (self.image_height, self.image_width)
        boxes = self.scale_bounding_box(boxes, input_shape, output_shape)

        # Convert the bounding box from (x, y, w, h) to (x1, y1, x2, y2)
        new_boxes = np.zeros_like(boxes)
        x_center, y_center, width, height = boxes.T

        new_boxes[:, 0] = x_center - width / 2  # Top-Left
        new_boxes[:, 1] = y_center - height / 2
        new_boxes[:, 2] = x_center + width / 2  # Bottom-Right
        new_boxes[:, 3] = y_center + height / 2

        # Clipping the bounding box
        new_boxes[:, 0] = np.clip(new_boxes[:, 0], 0, self.image_width)
        new_boxes[:, 1] = np.clip(new_boxes[:, 1], 0, self.image_height)
        new_boxes[:, 2] = np.clip(new_boxes[:, 2], 0, self.image_width)
        new_boxes[:, 3] = np.clip(new_boxes[:, 3], 0, self.image_height)

        return new_boxes

    def parse_masks(self, masks, boxes):
        # Get the probabilities for each pixel, then set the value either 0 or 255
        masks = sigmoid(masks)
        masks = (masks > 0.5).astype(np.uint8) * 255
        masks = masks.reshape((-1, self.mask_height, self.mask_width))

        # Scale the bounding box to the mask dimensions
        input_shape = (self.image_height, self.image_width)
        output_shape = (self.mask_height, self.mask_width)
        new_boxes = self.scale_bounding_box(boxes, input_shape, output_shape)

        # Process each box/mask pair
        new_masks = np.zeros((len(new_boxes), self.image_height, self.image_width))
        # blur_size = (int(self.image_width / self.mask_width), int(self.image_height / self.mask_height))

        for i in range(len(new_boxes)):
            # Get the coordinates of bounding box from mask dimensions and image dimensions
            scale_x1, scale_y1 = list(map(math.floor, new_boxes[i][:2]))
            scale_x2, scale_y2 = list(map(math.ceil, new_boxes[i][2:]))
            x1, y1 = list(map(math.floor, boxes[i][:2]))
            x2, y2 = list(map(math.ceil, boxes[i][2:]))

            # Resize the scaled crop mask to the original bounding box size
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

            # crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = cv2.medianBlur(crop_mask, ksize=7)
            # crop_mask = cv2.erode(crop_mask, np.ones((3, 3)), iterations=2)
            # crop_mask = cv2.dilate(crop_mask, np.ones((3, 3)), iterations=2)

            # Binarize the cropped mask
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            new_masks[i, y1:y2, x1:x2] = crop_mask

        return new_masks

    @staticmethod
    def scale_bounding_box(boxes, input_shape, output_shape):
        # Compute the scaler
        scale_x = output_shape[1] / input_shape[1]
        scale_y = output_shape[0] / input_shape[0]

        # Scale the bounding box
        new_boxes = boxes.copy()
        new_boxes[:, [0, 2]] *= scale_x
        new_boxes[:, [1, 3]] *= scale_y

        return new_boxes

    def process_NMS(self, boxes, confidences):
        """
        Apply Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes
        based on their confidence scores.
        """
        # Sort indices by confidence scores in descending order
        sorted_indices = np.argsort(confidences)[::-1]
        selected_indices = []

        while len(sorted_indices) > 0:
            # Select the box with the highest confidence score
            current_index = sorted_indices[0]
            selected_indices.append(current_index)

            if len(sorted_indices) == 1:
                break

            # Compute IoU of the selected box with the remaining boxes
            remaining_indices = sorted_indices[1:]
            ious = self.compute_IoU(boxes[current_index], boxes[remaining_indices])

            # Select boxes with IoU below the threshold
            below_threshold_indices = np.where(ious < self.min_iou)[0]

            # Update sorted_indices to keep only the boxes with IoU below the threshold
            sorted_indices = sorted_indices[below_threshold_indices + 1]

        return selected_indices

    def compute_IoU(self, box, boxes):
        """
        Compute the Intersection over Union (IoU) between a given box and a list of boxes.
        """
        # Compute the coordinates of the intersection rectangle
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute the intersection area
        intersection_width = np.maximum(0, xmax - xmin)
        intersection_height = np.maximum(0, ymax - ymin)
        intersection_area = intersection_width * intersection_height

        # Compute the union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute the IoU
        iou = intersection_area / union_area

        return iou

    def show(self, window_name: str = 'Segmentation'):
        # Draw the result
        result_image = self.draw_detections(
            self.predict_image,
            self._result_boxes,
            self._result_classes,
            self._result_confs,
            self._result_masks
        )

        # Show image
        cv2.imshow(window_name, result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_detections(self, image, boxes, classes, scores, masks, mask_alpha=0.3, mask_colors=None):
        # Set the font scale and text thickness based on image size
        image_height, image_width = image.shape[:2]
        text_size = min(image_height, image_width) * 0.0006
        text_thickness = int(min(image_height, image_width) * 0.001)

        output_image = image.copy()

        # If no mask colors are provided, generate random colors
        if mask_colors is None:
            mask_colors = random_colors(len(boxes))

        # Draw masks
        for i, (box, color) in enumerate(zip(boxes, mask_colors)):
            x1, y1, x2, y2 = box.astype(int)

            # Extract the mask region within the bounding box
            crop_mask = masks[i][y1:y2, x1:x2, np.newaxis]
            crop_output_image = output_image[y1:y2, x1:x2]

            # Blend the mask color with the image
            crop_output_image = crop_output_image * (1 - crop_mask) + crop_mask * color
            output_image[y1:y2, x1:x2] = crop_output_image

        # Apply the mask with the specified alpha value
        output_image = cv2.addWeighted(output_image, mask_alpha, image, 1 - mask_alpha, 0)

        # Draw bounding boxes and labels
        for box, score, class_, color in zip(boxes, scores, classes, mask_colors):
            color = color.tolist()
            x1, y1, x2, y2 = box.astype(int)

            # Draw the bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

            # Draw the label text
            label = class_names[class_]
            caption = f'{label} {int(score * 100)}%'
            (text_width, text_height), _ = cv2.getTextSize(text=caption,
                                                           fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                           fontScale=text_size,
                                                           thickness=text_thickness)
            text_height = int(text_height * 1.2)

            cv2.rectangle(output_image, (x1, y1), (x1 + text_width, y1 - text_height), color, -1)
            cv2.putText(output_image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        text_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return output_image

    def draw_detections_test(self, image, boxes, masks, mask_alpha=0.3, mask_colors=None):
        # Set the font scale and text thickness based on image size
        image_height, image_width = image.shape[:2]
        output_image = image.copy()

        # If no mask colors are provided, generate random colors
        if mask_colors is None:
            mask_colors = random_colors(len(boxes))

        # Draw masks
        for i, (box, color) in enumerate(zip(boxes, mask_colors)):
            x1, y1, x2, y2 = box.astype(int)

            # Extract the mask region within the bounding box
            crop_mask = masks[i][y1:y2, x1:x2, np.newaxis]
            crop_output_image = output_image[y1:y2, x1:x2]

            # Blend the mask color with the image
            crop_output_image = crop_output_image * (1 - crop_mask) + crop_mask * color
            output_image[y1:y2, x1:x2] = crop_output_image

        # Apply the mask with the specified alpha value
        output_image = cv2.addWeighted(output_image, mask_alpha, image, 1 - mask_alpha, 0)

        # Draw bounding boxes and labels
        for box, color in zip(boxes, mask_colors):
            color = color.tolist()
            x1, y1, x2, y2 = box.astype(int)

            # Draw the bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        return output_image

    def extract_instance(self, index):
        if not (0 <= index <= len(self.result_classes)):
            raise IndexError('The index is out of range, please check the target index.')

        # Get the selected bounding box and mask
        selected_box = self._result_boxes[index].astype(np.int32)
        selected_mask = self._result_masks[index]

        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = selected_box
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        # Apply the mask to the cropped image
        cropped_image = self.predict_image[y:y + h, x:x + w].copy()
        cropped_mask = selected_mask[y:y + h, x:x + w].copy().astype(np.bool_)

        cropped_image[~cropped_mask] = (0, 0, 0)

        return cropped_image
