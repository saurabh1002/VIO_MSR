import numpy as np
import rospy
import rosbag

from nav_msgs.msg import Odometry

dataset = np.loadtxt('poses_skip_15.txt')

with rosbag.Bag('/home/dhagash/NOBACKUP_CKA/skip_15.bag', 'w') as bag:
    for data in dataset:
        odom = Odometry()
        odom.header.frame_id = 'parent'
        t = str(data[0])
        odom.header.stamp.secs = int(t.split('.')[0])
        odom.header.stamp.nsecs = int(t.split('.')[1])
        odom.child_frame_id = 'imu_link'
        odom.pose.pose.position.x = data[1]
        odom.pose.pose.position.y = data[2]
        odom.pose.pose.position.z = data[3]
        odom.pose.pose.orientation.x = data[-4]
        odom.pose.pose.orientation.y = data[-3]
        odom.pose.pose.orientation.z = data[-2]
        odom.pose.pose.orientation.w = data[-1]
        odom.pose.covariance = np.diag([1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e-1]).ravel()
        odom.twist.twist.linear.x = 0
        odom.twist.twist.angular.z = 0
        odom.twist.covariance = np.diag([1e-2, 1e3, 1e3, 1e3, 1e3, 1e-2]).ravel()
        bag.write("/odom", odom, odom.header.stamp)
