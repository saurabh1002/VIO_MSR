import numpy as np
import rospy
import rosbag
import tf

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


def poses_to_rosbag(data_path: str, bag_path: str, topic_name: str, parent_frame:str, child_frame: str):
    with rosbag.Bag(bag_path, 'w') as bag:
        gt_data = np.loadtxt('/home/dhagash/tmp/VIO_MSR/datasets/phenorob/front/groundtruth.txt', delimiter=' ')
        dataset = np.loadtxt(data_path, delimiter=' ')
        cov = np.diag([1, 1, 1, 0.1, 0.5, 0.5]).ravel()
        for data in dataset:
            vo_odom = Odometry()
            vo_odom.header.frame_id = parent_frame
            t = str(data[0])
            vo_odom.header.stamp.secs = int(t.split('.')[0])
            vo_odom.header.stamp.nsecs = int(t.split('.')[1])
            vo_odom.child_frame_id = child_frame
            vo_odom.pose.pose.position.x = data[1]
            vo_odom.pose.pose.position.y = data[2]
            vo_odom.pose.pose.position.z = data[3]
            vo_odom.pose.pose.orientation.x = data[-4]
            vo_odom.pose.pose.orientation.y = data[-3]
            vo_odom.pose.pose.orientation.z = data[-2]
            vo_odom.pose.pose.orientation.w = data[-1]
            vo_odom.pose.covariance = cov
            vo_odom.twist.twist.linear.x = 0
            vo_odom.twist.twist.angular.z = 0
            vo_odom.twist.covariance = np.diag([1e-2, 1e3, 1e3, 1e3, 1e3, 1e-2]).ravel()

            bag.write(f"/{topic_name}", vo_odom, vo_odom.header.stamp)
    
        # for data in gt_data:
            # lidar_odom = Odometry()
            # lidar_odom.header.frame_id = 'map'
            # t = str(data[0])
            # lidar_odom.header.stamp.secs = int(t.split('.')[0])
            # lidar_odom.header.stamp.nsecs = int(t.split('.')[1])
            # lidar_odom.child_frame_id = 'lidar'
            # lidar_odom.pose.pose.position.x = data[1]
            # lidar_odom.pose.pose.position.y = data[2]
            # lidar_odom.pose.pose.position.z = data[3]
            # lidar_odom.pose.pose.orientation.x = data[-4]
            # lidar_odom.pose.pose.orientation.y = data[-3]
            # lidar_odom.pose.pose.orientation.z = data[-2]
            # lidar_odom.pose.pose.orientation.w = data[-1]
            # lidar_odom.pose.covariance = np.diag([1e-2, 1e-2, 1e-2, 1e3, 1e3, 1e-1]).ravel()
            # lidar_odom.twist.twist.linear.x = 0
            # lidar_odom.twist.twist.angular.z = 0
            # lidar_odom.twist.covariance = np.diag([1e-2, 1e3, 1e3, 1e3, 1e3, 1e-2]).ravel()
            # bag.write("/lidar_odom", lidar_odom, lidar_odom.header.stamp)
        

            tf_msg = TFMessage()
            # tf_cam_to_map = TransformStamped()
            # tf_cam_to_map.transform.translation.x = 0.0
            # tf_cam_to_map.transform.translation.y = 0.0
            # tf_cam_to_map.transform.translation.z = 0.0

            # quat = tf.transformations.quaternion_from_euler(-np.pi/2, np.pi/2, 0, 'szyx')
            
            # tf_cam_to_map.transform.rotation.x = quat[0]
            # tf_cam_to_map.transform.rotation.y = quat[1]
            # tf_cam_to_map.transform.rotation.z = quat[2]
            # tf_cam_to_map.transform.rotation.w = quat[3]

            # tf_cam_to_map.header.frame_id = 'map'
            # tf_cam_to_map.child_frame_id = parent_frame
            # tf_cam_to_map.header.stamp = lidar_odom.header.stamp



            tf_base_to_imu = TransformStamped()
            tf_base_to_imu.transform.translation.x = 0.0
            tf_base_to_imu.transform.translation.y = 0.0
            tf_base_to_imu.transform.translation.z = 0.0
            tf_base_to_imu.transform.rotation.x = 0.0
            tf_base_to_imu.transform.rotation.y = 0.0
            tf_base_to_imu.transform.rotation.z = 0.0
            tf_base_to_imu.transform.rotation.w = 1.0
            tf_base_to_imu.header.frame_id = 'base_link'
            tf_base_to_imu.child_frame_id = 'imu_link'
            tf_base_to_imu.header.stamp = vo_odom.header.stamp

            tf_map_to_odom = TransformStamped()
            tf_map_to_odom.transform.translation.x = 0.0
            tf_map_to_odom.transform.translation.y = 0.0
            tf_map_to_odom.transform.translation.z = 0.0
            tf_map_to_odom.transform.rotation.x = 0.0
            tf_map_to_odom.transform.rotation.y = 0.0
            tf_map_to_odom.transform.rotation.z = 0.0
            tf_map_to_odom.transform.rotation.w = 1.0
            tf_map_to_odom.header.frame_id = 'map'
            tf_map_to_odom.child_frame_id = 'odom'
            tf_map_to_odom.header.stamp = vo_odom.header.stamp

            tf_link_to_footprint = TransformStamped()
            tf_link_to_footprint.transform.translation.x = 0.0
            tf_link_to_footprint.transform.translation.y = 0.0
            tf_link_to_footprint.transform.translation.z = 0.0
            tf_link_to_footprint.transform.rotation.x = 0.0
            tf_link_to_footprint.transform.rotation.y = 0.0
            tf_link_to_footprint.transform.rotation.z = 0.0
            tf_link_to_footprint.transform.rotation.w = 1.0
            tf_link_to_footprint.header.frame_id = 'base_link'
            tf_link_to_footprint.child_frame_id = 'base_footprint'
            tf_link_to_footprint.header.stamp = vo_odom.header.stamp

            # tf_world_to_map = TransformStamped()
            # tf_world_to_map.transform.translation.x = 0.0
            # tf_world_to_map.transform.translation.y = 0.0
            # tf_world_to_map.transform.translation.z = 0.0
            # tf_world_to_map.transform.rotation.x = 0.0
            # tf_world_to_map.transform.rotation.y = 0.0
            # tf_world_to_map.transform.rotation.z = 0.0
            # tf_world_to_map.transform.rotation.w = 1.0
            # tf_world_to_map.header.frame_id = 'world'
            # tf_world_to_map.child_frame_id = 'map'
            # tf_world_to_map.header.stamp = vo_odom.header.stamp


            tf_base_to_cam = TransformStamped()
            tf_base_to_cam.transform.translation.x = 0.5
            tf_base_to_cam.transform.translation.y = 0.0
            tf_base_to_cam.transform.translation.z = 0.1
            tf_base_to_cam.transform.rotation.x = 0.7071
            tf_base_to_cam.transform.rotation.y = 0.7071
            tf_base_to_cam.transform.rotation.z = 0.0
            tf_base_to_cam.transform.rotation.w = 0.0
            tf_base_to_cam.header.frame_id = 'base_link'
            tf_base_to_cam.child_frame_id = 'cam_frame'
            tf_base_to_cam.header.stamp = vo_odom.header.stamp

            tf_msg.transforms = [tf_base_to_cam,tf_map_to_odom, tf_base_to_imu]

            bag.write("/tf", tf_msg, tf_msg.transforms[0].header.stamp)


if __name__ == '__main__':
    data_path = '/home/dhagash/Downloads/eval_data/front/apples_big_2021-10-14-all/superpoint/PnP/poses_min_dist_1_10deg.txt'
    bag_path = '/home/dhagash/tmp/VIO_MSR/eval_data/front/apples_big_2021-10-14-all/superpoint/PnP/poses_min_dist_1_10deg.bag'
    poses_to_rosbag(data_path, bag_path, 'vo', 'odom', 'cam_frame')