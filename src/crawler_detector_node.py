#!/usr/bin/env python

import sys, time
import numpy as np
import cv2
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from CrawlerDetector.api import CrawlerDetector

class CrawlerDetectorNode:
    def __init__(self):
        self.bridge = CvBridge()
        self._cam_sub = rospy.Subscriber("/robot/image_raw/", Image, self.callback, queue_size=1)
        self._cam_info_sub = rospy.Subscriber("/robot/camera_info", CameraInfo, self.cam_info_callback, queue_size=1)
        self._pose_pub = rospy.Publisher("/crawler/pose", PoseStamped, queue_size=1)
        self._do_display = rospy.get_param("/crawler_detector/do_display_detection") == True
	self._pose_tf = rospy.get_param("/crawler_detector/pose_tf")
	self._new_image = False
	self._detector = CrawlerDetector()

    def detect_crawler(self):
	if self._new_image:
	    hm, uv_max = self._detector.detect(self._image_np, is_bgr=True, do_display_detection=self._do_display)
	    self._publish_pose(uv_max)
	    self._new_image = False

    def cam_info_callback(self, msg):
           self._fx = msg.P[0]
           self._cx = msg.P[2]
           self._fy = msg.P[5]
           self._cy = msg.P[6]

    def callback(self, ros_data):
	try:
           self._image_np = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
	   self._new_image = True
        except CvBridgeError as e:
            print(e)


    def _publish_pose(self, uv):
        msg = PoseStamped()
        msg.header.frame_id = self._pose_tf
        msg.header.stamp = rospy.Time.now()
        msg.pose.orientation.w = 1.0
        if uv[0]!=-1 and uv[1]!=-1:
            msg.pose.position.z = 1.0
            msg.pose.position.x = (uv[0]-self._cx)/self._fx
            msg.pose.position.y = (uv[1]-self._cy)/self._fy
            self._pose_pub.publish(msg)

def main(args):
    node = CrawlerDetectorNode()

    rospy.init_node('crawler_detector', anonymous=True)
    while not rospy.is_shutdown():
	node.detect_crawler()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
