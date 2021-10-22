import os
import argparse

# parse command line
parser = argparse.ArgumentParser(description='''
This script takes a data folder with timestamps as name and generate a txt file with the associated timestamp and file names  
''')
parser.add_argument('data_dir', help='Path from root to the directory of the dataset')
parser.add_argument('output_filename', help='Name of txt file to store image data in the format: timestamp, imagename)')
args = parser.parse_args()

DATA_DIR = args.data_dir
op_filename = args.output_filename

f = open(op_filename,"w+")

for filename in os.listdir(DATA_DIR):
    time = filename.split('.p')[0]
    f.write("%f"%float(time) + " " + DATA_DIR + filename + "\n")

f.close()