import math
import os

import cv2
import numpy as np
from tqdm import tqdm

from capture import MediaCapture
from models import YOLOv8Seg


class ObjectRemover():
    def __init__(self):
        self.model_yolov8 = None
        self.model_propainter = None
        self.capture = None

        self._filename = None

        self._win_width = None
        self._win_height = None

        self._trajectory = []
        self._is_selected = False
        self._selected_index = None

    def load(self, filename: str):
        # Load the YOLOv8 segmentation model if not loaded already
        if self.model_yolov8 is None:
            self.model_yolov8 = YOLOv8Seg()
            self.model_yolov8.load_model(gpu=False)

        # Initialize media capture with the given filename
        self.capture = MediaCapture(filename, onstream=False)
        self._filename = os.path.splitext(os.path.split(filename)[1])[0]
        self._win_width = int(self.capture.width // 2)
        self._win_height = int(self.capture.height // 2)

    def select(self):
        if self.model_yolov8 is None:
            raise AttributeError('Model is not initialized. Call load() method first.')

        first_frame = None
        while True:
            # Retrieve the first frame
            while first_frame is None and self.capture.is_opened():
                first_frame = self.capture.read()

            if first_frame is None:
                raise AttributeError('Can not retrieve any frame from the stream.')

            # Perform instance segmentation on the first frame
            self.model_yolov8.predict(first_frame)
            first_frame = self.model_yolov8.draw_detections(
                self.model_yolov8.predict_image,
                self.model_yolov8.result_boxes,
                self.model_yolov8.result_classes,
                self.model_yolov8.result_confs,
                self.model_yolov8.result_masks
            )

            # Check whether there is any object existed
            if len(self.model_yolov8.result_classes) > 0:
                break
            first_frame = None

        # Create a window for user selection
        win_name = f'{__class__.__name__} - Selecting'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self._win_width, self._win_height)
        cv2.imshow(win_name, first_frame)
        cv2.setMouseCallback(win_name, self.__mouse_callback)

        # Update the displayed image based on user input
        while True:
            if not self._is_selected:
                result_image = self.draw_trajectory(first_frame)
            else:
                result_image = self.draw_selection(first_frame, self.model_yolov8.result_boxes)

            cv2.imshow(win_name, result_image)

            # Close the selection (ESC)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Keep showing the window until user select the object
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                if self._selected_index is not None:
                    break

                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, self._win_width, self._win_height)
                cv2.setMouseCallback(win_name, self.__mouse_callback)

        cv2.destroyAllWindows()

    def __mouse_callback(self, event, x, y, flags, param):
        # Mouse event callback for selecting the object trajectory
        if event == cv2.EVENT_LBUTTONDOWN:
            self._is_selected = False
            self._trajectory = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self._trajectory.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_selected = True
            self._trajectory.append((x, y))

    def draw_trajectory(self, image):
        # Draw the trajectory on the image
        result_image = image.copy()

        for i in range(1, len(self._trajectory)):
            cv2.line(result_image, self._trajectory[i - 1], self._trajectory[i], (0, 255, 0), 2)

        return result_image

    def draw_selection(self, image, boxes):
        result_image = image.copy()

        # Calculate bounding box of the trajectory
        trajectory = np.array(self._trajectory)
        x, y, w, h = cv2.boundingRect(trajectory)
        box = np.array([x, y, x + w, y + h])

        # Compute IoU between the bounding box and detected object
        ious = self.model_yolov8.compute_IoU(box, boxes)

        if np.all(ious == 0):  # No overlapping
            return result_image

        # Select the box with highest IoU and create a mask for it
        self._selected_index = np.argmax(ious)
        selected_box = boxes[self._selected_index].astype(np.int32)

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[selected_box[1]:selected_box[3], selected_box[0]:selected_box[2]] = (255, 255, 255)

        # Blend the mask with the original image
        result_image = cv2.addWeighted(result_image, 0.5, mask, 0.5, 0)

        return result_image

    def match(self, src_image, dst_image):
        # Predict instances in the destination image
        self.model_yolov8.predict(dst_image)

        # Initialize ORB detector and BFMatcher
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Detect keypoints and compute descriptors for the source image
        keypointsA, descriptorsA = orb.detectAndCompute(src_image, None)

        best_similarity = 0.35
        max_possible_matches = len(descriptorsA)
        
        matched_index = None
        matched_image = None

        # Iterate through each instance detected in the destination image
        for index in range(len(self.model_yolov8.result_classes)):
            # Extract the instance image from the destination image
            compared_image = self.model_yolov8.extract_instance(index)

            # Detect keypoints and compute descriptors for the instance image
            keypointsB, descriptorsB = orb.detectAndCompute(compared_image, None)

            if descriptorsB is None:
                continue

            # Match descriptors between the source and instance images
            matches = bf.match(descriptorsA, descriptorsB)
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = len(matches)

            # Update the best similarity and matched index/image if the current similarity is higher
            similarity = num_matches / max_possible_matches

            if similarity > best_similarity:
                best_similarity = similarity

                matched_index = index
                matched_image = compared_image

        return matched_index, matched_image

    def track(self, start_frame, end_frame, target_instance_image):
        moving_mask = self.model_yolov8.result_masks[self._selected_index].copy()

        # Initialize lists to record frames, bounding boxes, and masks
        record_frames = []
        record_boxes = []
        record_masks = []

        # Retrieve all frame
        self.capture.frame = start_frame - 1
        current_frame = start_frame - 1

        while self.capture.is_opened():
            # Retrieve current frame image
            frame = self.capture.read()

            if frame is None:
                self.capture.frame = start_frame
                break

            record_frames.append(frame)

            # Break if reach the end frame
            current_frame = current_frame + 1
            if current_frame > end_frame:
                self.capture.frame = start_frame
                break

        # Track the target instance on each frame
        progress = tqdm(record_frames, desc=f"Tracking", total=len(record_frames))

        for frame in progress:
            # Match the target instance with the current frame
            matched_index, matched_image = self.match(target_instance_image, frame)

            # Combining the instance moving mask
            if matched_index is not None:
                matched_mask = self.model_yolov8.result_masks[matched_index].copy()
                matched_box = self.model_yolov8.result_boxes[matched_index].copy()

                # Record the bounding box and mask
                x1, y1 = list(map(math.floor, matched_box[:2]))
                x2, y2 = list(map(math.floor, matched_box[2:]))

                record_boxes.append([x1, y1, x2, y2])
                record_masks.append(matched_mask[y1:y2, x1:x2])

                # Combine the new mask with the moving mask
                moving_mask = cv2.bitwise_or(moving_mask, matched_mask)

                # Update the target instance for the next frame
                target_instance_image = matched_image.copy()
            else:
                record_boxes.append(None)
                record_masks.append(None)

        return moving_mask, record_frames, record_boxes, record_masks

    def run(self, learning_base = False):
        # Select the target object to remove
        self.select()

        # Extract the selected image
        if self._selected_index is None:
            raise AttributeError('Object is not selected. Call select() method first.')
        selected_image = self.model_yolov8.extract_instance(self._selected_index)

        # Inpaint the image
        if learning_base:
            self.__inpaint_with_learning_base(selected_image)
        else:
            self.__inpaint_without_learning_base(selected_image)

    def __inpaint_without_learning_base(self, target_instance_image):
        # Tracking the target instance
        moving_mask, frames, boxes, masks = self.track(self.capture.frame - 1,
                                                       self.capture.total_frames,
                                                       target_instance_image)

        # Process each recorded bounding box and mask
        results = []

        progress = tqdm(enumerate(zip(frames, boxes, masks)), 
                        desc=f"Inpaint without learning base", total=len(frames))

        for current_index, (frame, box, mask) in progress:
            # Directly ignore the image that didn't catch any instance
            if box is None:
                results.append(frame)
                continue

            # Initialize a background image and a filled mask
            background_image = np.zeros((self.capture.height, self.capture.width, 3))
            filled_mask = moving_mask.copy()

            # Inpaint the background by previous or next frame image
            reached_start, reached_end = False, False
            offset = -1

            while not reached_start or not reached_end:
                offset = offset + 1

                # Inpainting based on previous frame
                prev_index = current_index - offset
                if prev_index < 0:
                    reached_start = True

                if not np.any(filled_mask):
                    break

                if not reached_start:
                    # Create a mask for the previous frame
                    x1, y1, x2, y2 = boxes[prev_index]
                    target_instance_mask = np.zeros((self.capture.height, self.capture.width))
                    target_instance_mask[y1:y2, x1:x2] = masks[prev_index]

                    # Copy the background pixels from the previous frame where the mask excludes the target instance
                    exclude_mask = cv2.bitwise_xor(filled_mask, target_instance_mask)
                    remain_mask = cv2.bitwise_and(filled_mask, exclude_mask).astype(np.bool_)
                    filled_mask = cv2.bitwise_and(filled_mask, target_instance_mask)
                    background_image[remain_mask] = frames[prev_index][remain_mask]

                # Inpainting based on next frame
                if not np.any(filled_mask):
                    break

                next_index = current_index + offset
                if next_index >= len(frames):
                    reached_end = True

                if not reached_end:
                    # Create a mask for the next frame
                    x1, y1, x2, y2 = boxes[next_index]
                    target_instance_mask = np.zeros((self.capture.height, self.capture.width))
                    target_instance_mask[y1:y2, x1:x2] = masks[next_index]
                    
                    # Copy the background pixels from the next frame where the mask excludes the target instance
                    exclude_mask = cv2.bitwise_xor(filled_mask, target_instance_mask)
                    remain_mask = cv2.bitwise_and(filled_mask, exclude_mask).astype(np.bool_)
                    filled_mask = cv2.bitwise_and(filled_mask, target_instance_mask)
                    background_image[remain_mask] = frames[next_index][remain_mask]

            background_image = background_image.astype(np.uint8)

            # Create a mask for the current frame
            x1, y1, x2, y2 = boxes[current_index]
            target_instance_mask = np.zeros((self.capture.height, self.capture.width))
            target_instance_mask[y1:y2, x1:x2] = masks[current_index]
            target_instance_mask = target_instance_mask.astype(np.bool_)

            # Replace the target instance in the current frame with the background
            frame[target_instance_mask] = background_image[target_instance_mask]
            results.append(frame.astype(np.uint8))

        # Display the output image and save it
        output = cv2.VideoWriter(f'./outputs/[Inpaint] {self._filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                                 self.capture.fps, (self.capture.width, self.capture.height))

        for frame in results:
            output.write(frame)

            win_name = f'{__class__.__name__} - Inpaint'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, self._win_width, self._win_height)
            cv2.imshow(win_name, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        output.release()

    def __inpaint_with_learning_base(self, target_instance_image):
        pass


def main():
    # filepath = './resources/IMG_1722.jpg'
    filepath = './resources/4K African Animals - Serengeti National Park.mp4'

    remover = ObjectRemover()
    remover.load(filepath)
    remover.run()


if __name__ == '__main__':
    main()
