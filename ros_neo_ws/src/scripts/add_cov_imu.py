from tokenize import Double
from parso import parse
import rosbag
import numpy as np
import argparse

from sensor_msgs.msg import Imu

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument('-t', '--topic', type=str, help='Topic name')
    parser.add_argument('-i', '--file_name',type=str, help='Bag file')

    args = parser.parse_args()
    
    BAGFILE = args.file_name
    TOPIC = args.topic

    with rosbag.Bag('/home/dhagash/NOBACKUP_CKA/apples_big_2021-10-14-all_cov.bag', 'w') as write_bag:
        bag = rosbag.Bag(BAGFILE)
        odom_topic = bag.read_messages("/imu/data")

        for k, b in enumerate(odom_topic):
            imu = Imu()

            imu.header = b.message.header
            imu.orientation = b.message.orientation
            imu.angular_velocity = b.message.angular_velocity
            imu.linear_acceleration = b.message.linear_acceleration

            imu.orientation_covariance = np.diag([1, 1, 0.1]).ravel()
            imu.linear_acceleration_covariance = np.diag([10, 10, 10]).ravel()
            imu.angular_velocity_covariance = np.diag([1, 1, 0.1]).ravel()

            write_bag.write("/imu/data", imu, imu.header.stamp)