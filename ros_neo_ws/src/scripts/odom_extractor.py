from tokenize import Double
from parso import parse
import rosbag
import numpy as np
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Dataset paths
    parser.add_argument('-t', '--topic', type=str, help='Topic name')
    parser.add_argument('-i', '--file_name',type=str, help='Bag file')
    parser.add_argument('-o', '--output_file', type=str, help='Outfile name')

    args = parser.parse_args()
    
    FILE_SAVE = args.output_file
    BAGFILE = args.file_name
    TOPIC = args.topic

    bag = rosbag.Bag(BAGFILE)
    odom_topic = bag.read_messages(TOPIC)
    # print(len(odom_topic))
    poses = []
    for k, b in enumerate(odom_topic):
        
        t = str(b.message.header.stamp.secs) + '.' + str(b.message.header.stamp.nsecs)
        t = float(t)
        # print(t)
        x = b.message.pose.pose.position.x
        y = b.message.pose.pose.position.y
        z = b.message.pose.pose.position.z

        qx = b.message.pose.pose.orientation.x
        qy = b.message.pose.pose.orientation.y
        qz = b.message.pose.pose.orientation.z
        qw = b.message.pose.pose.orientation.w

        poses.append([t, x, y, z, qx, qy, qz, qw])
        # print(poses)

    poses = np.array(poses, np.float64)

    np.savetxt(FILE_SAVE, poses)