import cv2
import pafy

from model import YOLOv8Seg
from capture import MediaCapture


def main():
    model = YOLOv8Seg()
    model.load_model(gpu=True)

    url = 'https://www.youtube.com/watch?v=L4U4woKIC_w&list=PLWLEgnHzuoYx8Y4s_HPV1H1EYmLT6X74E&index=85'
    # url = 'https://www.youtube.com/watch?v=gFRtAAmiFbE&list=PL-Ni-1OtjEdLtQRpD-6r9AsD3P_6MLpgv&index=96'
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")

    capture = MediaCapture(best.url, onstream=True)
    win_width = int(capture.width // 2)
    win_height = int(capture.height // 2)

    while True:
        frame = capture.read()

        if frame is None:
            continue

        model.predict(frame)
        output_image = model.draw_detections(
            model.predict_image,
            model.result_boxes,
            model.result_classes,
            model.result_confs,
            model.result_masks
        )

        win_name = 'Output'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, win_width, win_height)
        cv2.imshow(win_name, output_image)

        # Close the window (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
