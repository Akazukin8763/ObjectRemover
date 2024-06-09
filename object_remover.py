import math
import os

import cv2
import numpy as np
from tqdm import tqdm

from capture import MediaCapture
from models import Inpainter, YOLOv8Seg
from shadow_remover import find_contour_edges, process_shadow


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
            self.model_yolov8.load_model(gpu=True)

        # Load the Propainter model if not loaded already
        if self.model_propainter is None:
            self.model_propainter = Inpainter()
            self.model_propainter.load_model(gpu=True)

        # Initialize media capture with the given filename
        self.capture = MediaCapture(filename, onstream=False)
        self._filename = os.path.splitext(os.path.split(filename)[1])[0]
        self._win_width = int(self.capture.width)
        self._win_height = int(self.capture.height)

    def select(self):
        if self.model_yolov8 is None or self.model_propainter is None:
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

    def match(self, src_image, dst_image, src_box, max_distance=50):
        # Predict instances in the destination image
        self.model_yolov8.predict(dst_image)

        # Initialize ORB detector and BFMatcher
        orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=0)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Detect keypoints and compute descriptors for the source image
        keypointsA, descriptorsA = orb.detectAndCompute(src_image, None)

        best_similarity = 0.20
        max_possible_matches = len(descriptorsA)
        
        matched_index = None
        matched_image = None

        # Calculate the center point of the source box image
        src_center_x = (src_box[0] + src_box[2]) / 2
        src_center_y = (src_box[1] + src_box[3]) / 2

        # Iterate through each instance detected in the destination image
        for index in range(len(self.model_yolov8.result_classes)):
            # Extract the instance image from the destination image
            compared_image = self.model_yolov8.extract_instance(index)
            matched_box = self.model_yolov8.result_boxes[index]

            # Detect keypoints and compute descriptors for the instance image
            keypointsB, descriptorsB = orb.detectAndCompute(compared_image, None)

            if descriptorsB is None:
                continue

            # Match descriptors between the source and instance images
            matches = bf.match(descriptorsA, descriptorsB)
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = len(matches)

            # Update the best similarity and matched index/image if the current similarity is higher
            # Also, we need to ensure that the distance between two matched instances are not too far
            similarity = num_matches / max_possible_matches

            matched_center_x = (matched_box[0] + matched_box[2]) / 2
            matched_center_y = (matched_box[1] + matched_box[3]) / 2
            distance = math.sqrt((matched_center_x - src_center_x) ** 2 + (matched_center_y - src_center_y) ** 2)

            if similarity > best_similarity and distance <= max_distance:
                best_similarity = similarity

                matched_index = index
                matched_image = compared_image

        return matched_index, matched_image

    def match_shadow(self, target_instance_mask, dst_image):
        # Find all of the shadows
        shadow_size_threshold = int(self.capture.height * self.capture.width * 0.001)
        shadow_masks = process_shadow(dst_image, ab_threshold=0, shadow_size_threshold=shadow_size_threshold)

        # Match the shadow with target instance
        target_mask = cv2.dilate(target_instance_mask, np.ones((3, 3)), iterations=3)
        matched_mask = None

        target_edge_pixels = find_contour_edges(target_instance_mask)  # Get the target edges
        min_distance = min(self.capture.height, self.capture.width) * 0.05  # Threshold

        for shadow_mask in shadow_masks:
            # Calculate the overlap region ratio and relative size ratio
            overlap_area = np.logical_and(target_mask, shadow_mask)

            shadow_sum = np.sum(shadow_mask)
            overlap_sum = np.sum(overlap_area)
            target_sum = np.sum(target_mask)

            overlap_ratio = overlap_sum / shadow_sum if shadow_sum != 0 else 0
            relative_ratio = shadow_sum / target_sum if target_sum != 0 else 0

            # Ignore the candidated shadow which overlap half of the target instance, or the shadow is too large
            if overlap_ratio >= 0.5 or relative_ratio >= 1.5:
                continue

            # Update the best matched shadow mask if it is close to target
            shadow_edge_pixels = find_contour_edges(shadow_mask)  # Get the shadow edges

            distances = np.sqrt(
                ((target_edge_pixels[:, np.newaxis, :] - shadow_edge_pixels[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            distance = np.min(distances)

            if distance < min_distance:
                min_distance = distance
                matched_mask = shadow_mask

        return matched_mask

    def track(self, start_frame, end_frame, target_instance_image):
        moving_mask = self.model_yolov8.result_masks[self._selected_index].copy()

        # Initialize lists to record frames, bounding boxes, and masks
        record_frames = []
        record_boxes = []
        record_masks = []
        record_shadow_masks = []

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
        progress = tqdm(enumerate(record_frames), desc=f"Tracking", total=len(record_frames))

        # Add the box for matching frames
        src_box = self.model_yolov8.result_boxes[self._selected_index].copy()

        # Store the continuous masks, and use it to create a big masks
        mask_stack = []
        position_stack = np.empty((0, 5))
        delete_threshold = 10  # 3 for zebra scene, 10 for bird scene

        for id, frame in progress:
            # Match the target instance with the current frame
            matched_index, matched_image = self.match(target_instance_image, frame, src_box)

            # Combining the instance moving mask
            if matched_index is not None:
                matched_mask = self.model_yolov8.result_masks[matched_index].copy()
                matched_box = self.model_yolov8.result_boxes[matched_index].copy()

                # Record the bounding box and mask
                x1, y1 = list(map(math.floor, matched_box[:2]))
                x2, y2 = list(map(math.ceil, matched_box[2:]))

                position_stack = np.vstack((position_stack, np.array([id, x1, y1, x2, y2])))
                mask_stack.append(matched_mask)

                # Combine the new mask with the moving mask
                moving_mask = cv2.bitwise_or(moving_mask, matched_mask)

                # Update the target instance for the next frame
                target_instance_image = matched_image.copy()
                
                # Update the source box for the next frame
                src_box = matched_box.copy()

            # Get the maximum box sixe
            if len(position_stack):
                final_x1 = np.min(position_stack[:, 1]).astype(np.int16)
                final_y1 = np.min(position_stack[:, 2]).astype(np.int16)
                final_x2 = np.max(position_stack[:, 3]).astype(np.int16)
                final_y2 = np.max(position_stack[:, 4]).astype(np.int16)

                final_mask = np.zeros_like(matched_mask)

                for mask in mask_stack:
                    final_mask = cv2.bitwise_or(final_mask, mask)

                record_boxes.append([final_x1, final_y1, final_x2, final_y2])
                record_masks.append(final_mask[final_y1:final_y2, final_x1:final_x2])

                # Remove the box and mask that too old
                if id - position_stack[0][0] > delete_threshold:
                    position_stack = position_stack[1:]
                    mask_stack = mask_stack[1:]
            else:
                record_boxes.append(None)
                record_masks.append(None)

            # Find the shadow of target instance if matched
            if matched_index is not None:
                matched_mask = self.model_yolov8.result_masks[matched_index].copy()
                shadow_mask = self.match_shadow(matched_mask, frame)

                # Record the shadow mask
                record_shadow_masks.append(shadow_mask)
            else:
                record_shadow_masks.append(None)

        return moving_mask, record_frames, record_boxes, record_masks, record_shadow_masks

    def run(self, learning_base = False):
        # Select the target object to remove
        self.select()

        # Extract the selected image
        if self._selected_index is None:
            raise AttributeError('Object is not selected. Call select() method first.')
        selected_image = self.model_yolov8.extract_instance(self._selected_index)

        # Inpaint the image
        if learning_base:
            results = self.__inpaint_with_learning_base(selected_image)
        else:
            results = self.__inpaint_without_learning_base(selected_image)

        # Create the outputs folder if it doesn't exist
        os.makedirs('outputs', exist_ok=True)

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

    def __inpaint_without_learning_base(self, target_instance_image):
        # Tracking the target instance
        moving_mask, frames, boxes, masks, shadow_masks = self.track(self.capture.frame - 1,
                                                                     self.capture.total_frames,
                                                                     target_instance_image)

        # Initialize the 5x5 kernel for dilation
        dilate_kernel = np.ones((11, 11), dtype=np.uint8)

        # Process each recorded bounding box and mask
        results = []

        progress = tqdm(enumerate(zip(frames, boxes, masks, shadow_masks)), 
                        desc=f"Inpaint without learning base", total=len(frames))

        for current_index, (frame, box, mask, shadow_mask) in progress:
            # Directly ignore the image that didn't catch any instance
            if box is None:
                results.append(frame)
                continue

            # Initialize a background image and a filled mask
            background_image = frame.copy()
            filled_mask = moving_mask.copy()

            if shadow_mask is not None:
                filled_mask = cv2.bitwise_or(filled_mask, shadow_mask)

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
                
                if not reached_start and boxes[prev_index] is not None:
                    # Create a mask for the previous frame
                    x1, y1, x2, y2 = boxes[prev_index]
                    target_instance_mask = np.zeros((self.capture.height, self.capture.width))
                    target_instance_mask[y1:y2, x1:x2] = masks[prev_index]

                    # Copy the background pixels from the previous frame where the mask excludes the target instance
                    exclude_mask = cv2.bitwise_xor(filled_mask, target_instance_mask)
                    remain_mask = cv2.bitwise_and(filled_mask, exclude_mask).astype(np.bool_)
                    filled_mask = cv2.bitwise_and(filled_mask, target_instance_mask)
                    background_image[remain_mask] = frames[prev_index][remain_mask]

                if not reached_start and boxes[prev_index] is not None:
                    # Create a mask for the previous frame
                    x1, y1, x2, y2 = boxes[prev_index]
                    target_instance_mask = np.zeros((self.capture.height, self.capture.width))
                    target_instance_mask[y1:y2, x1:x2] = masks[prev_index]
                    
                    # Dilate the mask 
                    target_instance_mask = cv2.dilate(target_instance_mask, dilate_kernel, iterations=1)

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

                if not reached_end and boxes[next_index] is not None:
                    # Create a mask for the next frame
                    x1, y1, x2, y2 = boxes[next_index]
                    target_instance_mask = np.zeros((self.capture.height, self.capture.width))
                    target_instance_mask[y1:y2, x1:x2] = masks[next_index]
                    
                    # Dilate the mask
                    target_instance_mask = cv2.dilate(target_instance_mask, dilate_kernel, iterations=1)

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

            # Dilate the mask
            target_instance_mask = cv2.dilate(target_instance_mask, dilate_kernel, iterations=1)
            target_instance_mask = target_instance_mask.astype(np.bool_)

            # Replace the target instance in the current frame with the background
            frame[target_instance_mask] = background_image[target_instance_mask]
            results.append(frame.astype(np.uint8))

        return results

    def __inpaint_with_learning_base(self, target_instance_image, window_size=80):
        # Tracking the target instance
        _, frames, boxes, masks, shadow_masks = self.track(self.capture.frame - 1,
                                                           self.capture.total_frames,
                                                           target_instance_image)

        # Create the original masks
        for i in range(len(boxes)):
            target_instance_mask = np.zeros((self.capture.height, self.capture.width))

            if boxes[i] is not None:
                x1, y1, x2, y2 = boxes[i]
                target_instance_mask[y1:y2, x1:x2] = masks[i]

                # Inpaint the instance with shadow
                if shadow_masks[i] is not None:
                    target_instance_mask = cv2.bitwise_or(target_instance_mask, shadow_masks[i])

            masks[i] = target_instance_mask

        # Use ProPainter to inpaint the target instance
        fragments = math.ceil(len(frames) / window_size)

        frames_split = np.array_split(frames, fragments)
        masks_split = np.array_split(masks, fragments)

        results = []
        for frame_chunk, mask_chunk in zip(frames_split, masks_split):
            results.extend(self.model_propainter.predict(frame_chunk, mask_chunk))

        return results


def main():
    # filepath = './resources/IMG_1722.jpg'
    filepath = './resources/640x360_Zebra.mp4'
    # filepath = './resources/640x360_Birds.mp4.mp4'
    # filepath = './resources/640x360_Eagle.mp4'
    # filepath = './resources/640x360_Giraffe.mp4'

    remover = ObjectRemover()
    remover.load(filepath)
    remover.run(learning_base=False)


if __name__ == '__main__':
    main()
