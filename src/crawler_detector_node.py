#!/usr/bin/env python

import sys, time
import numpy as np
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from CrawlerDetector.api import CrawlerDetector

class CrawlerDetectorNode:
    def __init__(self):
        self._detector = CrawlerDetector()
        self._cam_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback, queue_size=1)
        self._pose_pub = rospy.Publisher("/crawler/pose", PoseStamped, queue_size=10)

    def callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        hm, uv_max = self._detector.detect(image_np, is_bgr=True, do_display_detection=True)
        self._publish_pose(uv_max)

    def _publish_pose(self, uv):
        msg = PoseStamped()
        msg.header.frame_id = "/base_link"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.z = 0
        msg.pose.position.x = uv[0]
        msg.pose.position.y = uv[1]
        msg.pose.orientation.w = 1.0
        self._pose_pub.publish(msg)

def main(args):
    node = CrawlerDetectorNode()
    rospy.init_node('crawler_detector', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down crawler detector"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
