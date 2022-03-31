import os
import argparse
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--path",help="Path to labels")
    parser.add_argument("-o","--outpath",help="Path for output images")
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.outpath,"detections")):
        os.makedirs(os.path.join(args.outpath,"detections"))
    
    data = {}
    w = 480
    h = 640
    for names in os.listdir(args.path):
        img_name = names.split('.t')[0] 
        boxes = []
        with open(os.path.join(args.path,names),'r') as f:
            lines = f.readlines()
                
            for line in lines:
                if line == 'None':
                    print(names)
                    break
                xywh = line.split(" ")
                x_center = round(float(xywh[1]) * h)
                y_center = round(float(xywh[2]) * w)
                width = round(float(xywh[3])*h)
                height= round(float(xywh[4])* w)
                boxes.append([x_center,y_center,width,height])
            f.close()
        data[img_name] = boxes
    
    pickle.dump(data,open(os.path.join(os.path.join(args.outpath,"detections"),"detection.pickle"),'wb'))