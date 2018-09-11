import cv2
import datetime
import torch
import torchvision.transforms as transforms
from options.test_options import TestOptions
from models.models import ModelsFactory
import numpy as np

class CrawlerDetector:
    def __init__(self, prob_threshold=0.5, do_filter_prob=True, sliding_window_jump=10):
        self._prob_treshold = prob_threshold
        self._do_filter_prob = do_filter_prob
        self._detected_in_previous_frame = False
        self._slw_jump = sliding_window_jump

        self._opt = TestOptions().parse()  # parse model parameters
        self._img2tensor = self._create_img_transform()  # map RGB cv2 image to Pytorch tensor
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)  # get model
        self._model.set_eval()  # set model in test mode
        self._last_position = (-1, -1)

    def track(self, frame, is_bgr=False, do_display_detection=False):
        # search in last place
        if self._last_position != (-1, -1):
            hm, uv_max = self.detect(frame, is_bgr, do_display_detection, self._last_position)

        # search all frame
        else:
            for patch in self._yield_patches_from_frame(frame):
                hm, uv_max = self.detect(patch, is_bgr, do_display_detection)
                if uv_max != (-1, -1):
                    break

        self._last_position = uv_max
        return hm, uv_max


    def detect(self, frame, is_bgr=False, do_display_detection=False, center=None):
        # detect crawler
        crop_frame, frame_tensor, uv_top = self._preprocess_frame(frame, is_bgr, center)
        hm, uv_max, prob, elapsed_time = self._detect_crawler(frame_tensor)

        # filter prob
        final_prob = self._filter_prob(prob) if self._do_filter_prob else prob

        if prob is None or prob <= self._prob_treshold:
            uv_max = (-1, -1)

        else:
            # display detection
            if do_display_detection:
                self._display_center(crop_frame, uv_max, final_prob, elapsed_time)

            # update prob filter
            if self._do_filter_prob:
                self._update_prob_filter(prob)

            # add offset
            uv_max[0] += uv_top[0]
            uv_max[1] += uv_top[1]

        return hm, uv_max

    def _preprocess_frame(self, frame, is_bgr, center=None):
        # resize frame to half
        frame = cv2.resize(frame, (self._opt.image_size_w, self._opt.image_size_h))

        # crop frame to network size
        if center is None:
            crop_top = int((self._opt.image_size_h-self._opt.net_image_size)/2.0)
            crop_left = int((self._opt.image_size_w - self._opt.net_image_size) / 2.0)
        else:
            crop_top, crop_left = self._top_coords_crop_with_center(center)
        crop_frame = frame[crop_top:crop_top + self._opt.net_image_size, crop_left:crop_left + self._opt.net_image_size]

        # convert to pytorch tensor and add batch dimension
        frame_tensor = crop_frame.copy()
        if is_bgr:
            frame_tensor = cv2.cvtColor(frame_tensor, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.unsqueeze(self._img2tensor(frame_tensor), 0)

        return crop_frame, frame_tensor, (crop_left, crop_top)

    def _yield_patches_from_frame(self, frame):
        limit_i = self._opt.image_size_h-self._opt.net_image_size
        limit_j = self._opt.image_size_w-self._opt.net_image_size
        img_size = self._opt.net_image_size
        for i in range(0, limit_i, self._slw_jump):
            for j in range(0, limit_j, self._slw_jump):
                yield frame[i:i+img_size, j:j+img_size]

    def _top_coords_crop_with_center(self, center):
        u, v = center[0], center[1]
        rad = self._opt.net_image_size/2
        top_u = np.clip(u, rad, self._opt.image_size_w - rad) - rad
        top_v = np.clip(u, rad, self._opt.image_size_h - rad) - rad
        return (top_u, top_v)

    def _detect_crawler(self, frame_tensor):
        # bb as (top, left, bottom, right)
        start_time = datetime.datetime.now()
        hm, uv_max, prob = self._model.test(frame_tensor, do_normalize_output=False)
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        if hm is not None:
            hm = hm[0]
        return hm, uv_max, prob, elapsed_time

    def _display_bb(self, frame, bb, prob, elapsed_time):
        color = self._get_display_color(self._is_pos_detection(prob))

        # display bb
        (top, left, bottom, right) = bb
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # display prob
        prob_txt = '%.2f' % prob
        cv2.rectangle(frame, (left, bottom - 17), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, prob_txt, (left + 6, bottom - 3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # display detection time
        h, w, _ = frame.shape
        detection_time_txt = 'Detection Time[s]: %.3f' % elapsed_time
        cv2.putText(frame, detection_time_txt, (w-200, h-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,0), 1)

        # display frame
        cv2.imshow('Crawler Detector BB', frame)

    def _display_hm(self, frame, hm, uv_max, prob, elapsed_time, prob_threshold=0.5):
        # display hm
        hm = (np.transpose(hm, (1, 2, 0)) * 255).astype(np.uint8)
        # if prob is not None and prob < prob_threshold:
        #     hm = np.ones(hm.shape, dtype=hm.dtype)
        hm_img = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, hm_img, 0.3, 0)

        #display uv_max
        # frame = cv2.circle(frame, uv_max, 3, (255, 255, 0))

        # display detection time
        h, w, _ = frame.shape
        detection_time_txt = 'Detection Time[s]: %.3f' % elapsed_time
        cv2.putText(frame, detection_time_txt, (w - 200, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        # prob
        if prob is not None:
            cv2.putText(frame, '%.2f' % prob, (w - 200, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        # display frame
        cv2.imshow('Crawler Detector HM', frame)

    def _display_center(self, frame, uv_max, prob, elapsed_time):
        h, w, _ = frame.shape
        detection_time_txt = 'Detection Time[s]: %.3f' % elapsed_time
        cv2.putText(frame, detection_time_txt, (w - 200, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        if prob is not None and prob >= self._prob_treshold:
            frame = cv2.circle(frame, (uv_max[1], uv_max[0]), 3, (0, 0, 255), -1)

        # prob
        if prob is not None:
            cv2.putText(frame, '%.2f' % prob, (w - 200, h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

        # display frame
        cv2.imshow('Crawler Detector HM', frame)
        cv2.waitKey(1)

    def _create_img_transform(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        return transforms.Compose(transform_list)

    def _is_pos_detection(self, prob):
        return prob >= self._opt.classifier_threshold

    def _get_display_color(self, is_pos=True):
        return (0, 255, 0) if is_pos else (0, 0, 255)

    def _filter_prob(self, prob):
        if prob >= self._prob_treshold:
            return prob if self._detected_in_previous_frame else -1
        else:
            return prob

    def _update_prob_filter(self, prob):
        self._detected_in_previous_frame = prob >= self._prob_treshold
