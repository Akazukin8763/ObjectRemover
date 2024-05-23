import cv2
import numpy as np

from capture import MediaCapture
from model import YOLOv8Seg


class ObjectRemover():
    def __init__(self):
        self.model = None
        self.capture = None

        self._win_width = None
        self._win_height = None

        self._trajectory = []
        self._is_selected = False
        self._selected_index = None

    def load(self, filename: str):
        # Load the YOLOv8 segmentation model if not loaded already
        if self.model is None:
            self.model = YOLOv8Seg()
            self.model.load_model(gpu=False)

        # Initialize media capture with the given filename
        self.capture = MediaCapture(filename, onstream=False)
        self._win_width = int(self.capture.width // 2)
        self._win_height = int(self.capture.height // 2)

    def select(self):
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load() method first.')

        first_frame = None
        while True:
            # Retrieve the first frame
            while first_frame is None and self.capture.is_opened():
                first_frame = self.capture.read()

            if first_frame is None:
                raise AttributeError('Can not retrieve any frame from the stream.')

            # Perform instance segmentation on the first frame
            self.model.predict(first_frame)
            first_frame = self.model.draw_detections(
                self.model.predict_image,
                self.model.result_boxes,
                self.model.result_classes,
                self.model.result_confs,
                self.model.result_masks
            )

            # Check whether there is any object existed
            if len(self.model.result_classes) > 0:
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
                result_image = self.draw_selection(first_frame, self.model.result_boxes)

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
        ious = self.model.compute_IoU(box, boxes)

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
        self.model.predict(dst_image)

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
        for index in range(len(self.model.result_classes)):
            # Extract the instance image from the destination image
            compared_image = self.model.extract_instance(index)

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

    def run(self):
        # Select the target object to remove
        self.select()

        # Extract the selected image
        if self._selected_index is None:
            raise AttributeError('Object is not selected. Call select() method first.')
        selected_image = self.model.extract_instance(self._selected_index)

        # Track the selected image on each frame
        while self.capture.is_opened():
            # Retrieve current frame image
            frame = self.capture.read()

            if frame is None:
                break

            # Match the selected image with the current frame
            matched_index, matched_image = self.match(selected_image, frame)

            # Generate output image based on the match result
            if matched_index is None:
                output_image = frame
            else:
                output_image = self.model.draw_detections(
                    self.model.predict_image,
                    [self.model.result_boxes[matched_index]],
                    [self.model.result_classes[matched_index]],
                    [self.model.result_confs[matched_index]],
                    [self.model.result_masks[matched_index]]
                )
                selected_image = matched_image.copy()

            # Display the output image
            win_name = f'{__class__.__name__} - Tracking'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, self._win_width, self._win_height)
            cv2.imshow(win_name, output_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


def main():
    # filepath = './resources/IMG_1722.jpg'
    filepath = './resources/4K African Animals - Serengeti National Park.mp4'

    remover = ObjectRemover()
    remover.load(filepath)
    remover.run()


if __name__ == '__main__':
    main()
