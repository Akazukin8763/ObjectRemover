import cv2
import threading


image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']


class MediaCapture:
    def __init__(self, filename, onstream: bool = False):
        self._is_image = any(filename.lower().endswith(ext) for ext in image_extensions)
        self._is_read = False

        if self._is_image:
            self.image = cv2.imread(filename)
            self._width = self.image.shape[1]
            self._height = self.image.shape[0]
            self._fps = 0
        else:
            self.onstream = onstream
            self.capture = cv2.VideoCapture(filename)

            self._width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = int(self.capture.get(cv2.CAP_PROP_FPS))

            self.lock = threading.Lock()
            self.event = threading.Event()

            if self.onstream:
                self.thread = threading.Thread(target=self._reader)
                self.thread.daemon = True
                self.thread.start()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def fps(self):
        return self._fps
    
    @property
    def frame(self):
        if not self._is_image:
            return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return 0

    @frame.setter
    def frame(self, new_frame: int):
        if not self._is_image:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    def _reader(self):
        while not self.event.is_set():
            try:
                with self.lock:
                    ret = self.capture.grab()
                if not ret:
                    break
            except Exception:
                pass

    def is_opened(self):
        if self._is_image:
            return not self._is_read
        else:
            return self.capture.isOpened()

    def read(self):
        if self._is_image:
            self._is_read = True
            return self.image
        else:
            if self.onstream:
                with self.lock:
                    _, frame = self.capture.retrieve()
            else:
                _, frame = self.capture.read()
            return frame

    def release(self):
        if not self._is_image:
            self.event.set()
            self.capture.release()
