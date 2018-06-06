import cv2
from api import CrawlerDetector

class DetectCamNoRos:
    def __init__(self, process_every_n=3):
        self._video_capture = cv2.VideoCapture(0)
        self._crawler_detector = CrawlerDetector()
        self._process_every_n = process_every_n
        self._run()

    def _run(self):
        n_to_process = self._process_every_n

        while True:
            ret, frame = self._video_capture.read()
            n_to_process -= 1

            if ret and n_to_process == 0:
                self._detect_crawler(frame)
                n_to_process = self._process_every_n

            # Quit with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._video_capture.release()
        cv2.destroyAllWindows()

    def _detect_crawler(self, frame):
        return self._crawler_detector.detect(frame, is_bgr=True, do_display_detection=True)



if __name__ == '__main__':
    DetectCamNoRos()