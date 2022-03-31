import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import argparse





TOPIC = '/imu/data'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument('-i', '--file_name',type=str, help='Bag file')
    parser.add_argument('-o', '--output_file', type=str, help='Outfile name')

    args = parser.parse_args()
    
    FILE_SAVE = args.output_file
    BAGFILE = args.file_name
    bag = rosbag.Bag(BAGFILE)
    imu_topic = bag.read_messages(TOPIC)
    f = open(FILE_SAVE,'a+')
    for k, b in enumerate(imu_topic):
        timestamp = str(b.message.header.stamp.secs) + '.' + str(b.message.header.stamp.nsecs)
        
        ang_x = str(b.message.angular_velocity.x)
        ang_y = str(b.message.angular_velocity.y)
        ang_z = str(b.message.angular_velocity.z)

        acc_x = str(b.message.linear_acceleration.x)
        acc_y = str(b.message.linear_acceleration.y)
        acc_z = str(b.message.linear_acceleration.z)

        f.write(f"{timestamp} {ang_x} {ang_y} {ang_z} {acc_x} {acc_y} {acc_z}\n")

    f.close()

    bag.close()

    # print('PROCESS COMPLETE')
