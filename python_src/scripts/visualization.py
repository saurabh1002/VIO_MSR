from ast import arg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''VO using PnP algorithm with Superpoint Features''')
    # Dataset paths
    parser.add_argument('-i', '--text_file_path',type=str, help='Path to the root directory of the dataset')
    parser.add_argument('-v', '--visualize_z', default=False, type=bool, help="Visulaize z axis")
    args = parser.parse_args()
    gt = np.loadtxt('/home/dhagash/tmp/VIO_MSR/datasets/phenorob/front/groundtruth.txt')
    best_plot = np.loadtxt('/home/dhagash/Downloads/eval_data/front/apples_big_2021-10-14-all/superpoint/PnP/poses_min_dist_1_10deg.txt')
    imu = np.loadtxt('/home/dhagash/tmp/VIO_MSR/eval_data/front/apples_big_2021-10-14-all/superpoint/PnP/poses_combined_min_dist_1_10deg.txt')
    POSE_PATH = args.text_file_path
    FILENAME = POSE_PATH.split('/')[-1].split('.')[0]
    vo_poses = np.loadtxt(POSE_PATH)
    T_rot =  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    vo_poses = vo_poses[:,1:4] @ T_rot
    best_plot = best_plot[:,1:4] @ T_rot
    imu_plot = imu[:,1:4] @ T_rot

    x = vo_poses[:, 0]
    y = vo_poses[:, 1]
    z = vo_poses[:, 2]

    gt_x = gt[:, 1]
    gt_y = gt[:, 2]
    gt_z = gt[:, 3]

    best_x = best_plot[:-3,0]
    best_y = best_plot[:-3,1]
    best_z = best_plot[:-3,2]

    imu_x = imu_plot[:-200,0]
    imu_y = imu_plot[:-200,1]
    imu_z = imu_plot[:-200,2]


    mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 2
    cmap = cm.get_cmap('Dark2')

    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_axes([0, 0, 0.6, 0.4])
    ax1.plot(best_x, best_y,linewidth=2,color= cmap(0),label='superpoint')
    ax1.plot(imu_x, imu_y,linewidth=2,color= cmap(1),label='superpoint+IMU')
    # ax1.plot(best_x, best_y,linewidth=2,color= cmap(1),label='superpoint (long-baseline)')
    # ax1.plot(x, y,linewidth=2,color= cmap(0),label='superpoint')
    ax1.plot(gt_x, gt_y, linewidth = 2, color = cmap(2),label = 'liosam')
    # ax1.plot(gt_x, gt_y, linewidth = 2, color = cmap(2),label = 'liosam')
    ax1.axis('equal')
    ax1.legend()
    ax1.set_xlabel('X (m)', labelpad=5)
    ax1.set_ylabel('Y (m)', labelpad=3)
    # plt.savefig(f'../../eval_data/plots/trajectory_{FILENAME}.png', dpi=1000, transparent=True, bbox_inches='tight')
    # plt.savefig(f'../../eval_data/plots/trajectory_comparison.png', dpi=1000, transparent=True, bbox_inches='tight')
    plt.savefig(f'../../eval_data/plots/trajectory_comparison_imu.png', dpi=1000, transparent=True, bbox_inches='tight')

    plt.show()

    # if args.visualize_z:

    #     fig = plt.figure(figsize=(12,8))
    #     ax = fig.add_axes([0, 0, 0.6, 0.4])
    #     ax.plot(x, z,linewidth=2,color= cmap(0),label='superpoint')
    #     ax.plot(gt_x, gt_z,linewidth=2,color= cmap(1),label='liosam')
    #     ax.axis('equal')
    #     ax.set_xlim(0, 80)
    #     ax.set_ylim(-5, 3)
    #     # ax.set_aspect('equal', 'datalim',anchor='NE')
    #     ax.legend(loc='lower left')
    #     ax.set_xlabel('X (m)', labelpad=5)
    #     ax.set_ylabel('Z (m)', labelpad=3)
    #     # ax.set_aspect('equal', 'datalim',anchor='NE')
    #     ax.legend(loc='lower left')
    #     ax.set_xlabel('X (m)', labelpad=5)
    #     ax.set_ylabel('Z (m)', labelpad=3)
    #     plt.savefig(f'../../eval_data/plots/z_drift_{FILENAME}.png', dpi=1000, transparent=True, bbox_inches='tight')
    #     # plt.show()
