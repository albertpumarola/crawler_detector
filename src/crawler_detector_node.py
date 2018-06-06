#!/usr/bin/env python

import sys, time
import numpy as np
import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from CrawlerDetector.api import CrawlerDetector

class CrawlerDetectorNode:
    def __init__(self):
        self._detector = CrawlerDetector()
        self.bridge = CvBridge()
        self._cam_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, self.callback, queue_size=1)
        self._pose_pub = rospy.Publisher("/crawler/pose", PoseStamped, queue_size=1)
	self._new_image = False

    def detect_crawler(self):
	if self._new_image:
	    hm, uv_max = self._detector.detect(self._image_np, is_bgr=True, do_display_detection=True)
	    self._publish_pose(uv_max)
	    self._new_image = False

    def callback(self, ros_data):
	try:
           self._image_np = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
	   self._new_image = True
        except CvBridgeError as e:
            print(e)

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
    while not rospy.is_shutdown():
	node.detect_crawler()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
