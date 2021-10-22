import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np



FILENAME = 'apples_2021-10-14-14-36-15_0'
ROOT_DIR = '/home/dhagash/Project-2/dataset_orb/mapping'
DEPTH_SAVE = '/home/dhagash/Project-2/dataset_orb/mapping/images_apples/depth/'
COLOR_SAVE = '/home/dhagash/Project-2/dataset_orb/mapping/images_apples/rgb/'
BAGFILE = ROOT_DIR + '/' + FILENAME + '.bag'

if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    for i in range(2):
        if (i == 0):
            TOPIC = '/d455_front/depth/image_rect_raw'
            DESCRIPTION = 'depth_'
        else:
            TOPIC = '/d455_front/color/image_raw'
            DESCRIPTION = 'color_'
        image_topic = bag.read_messages(TOPIC)
        for k, b in enumerate(image_topic):
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
            cv_image.astype(np.uint8)
            if (DESCRIPTION == 'depth_'):
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(DEPTH_SAVE  + str(b.timestamp.secs) + '.' + str(b.timestamp.nsecs) + '.png', cv_image)
                
            else:
                cv2.imwrite(COLOR_SAVE + str(b.timestamp.secs) + '.' + str(b.timestamp.nsecs) + '.png', cv_image)
            print('saved: ' + DESCRIPTION + str(b.timestamp.secs) + '.' + str(b.timestamp.nsecs) + '.png')


    bag.close()

    print('PROCESS COMPLETE')