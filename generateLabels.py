import os
from tqdm import tqdm
import shutil
# path = "/home/krf/datasets/BDD/train/night/"
# labelpath = "/home/krf/datasets/BDD/train/yolo_night/labels/"
# imagepath = "/home/krf/datasets/BDD/train/yolo_night/images/"

# path = "/home/krf/datasets/BDD/test/night/"
# labelpath = "/home/krf/datasets/BDD/test/yolo_night/labels/"
# imagepath = "/home/krf/datasets/BDD/test/yolo_night/images/"

path = "/home/krf/datasets/BDD/test/night/"
labelpath = "/home/krf/datasets/BDD/test/yolo_night/labels/"
imagepath = "/home/krf/datasets/BDD/test/yolo_night/images/"
def move_images():
    flist = os.listdir(path)
    os.makedirs(imagepath,  exist_ok=True)
    for f in tqdm(flist):
        # 979dd162-267ea7e6.car_box2d
        fname = f.split('.')[0]
        ext = f.split('.')[1]
        if ext == "jpg":
            shutil.copyfile(path + f, imagepath + f)


def gen_labels():
    flist = os.listdir(path)
    os.makedirs(labelpath,  exist_ok=True)
    for f in tqdm(flist):
        # 979dd162-267ea7e6.car_box2d
        fname = f.split('.')[0]
        ext = f.split('.')[1]
        if ext == "car_box2d":
            content = open(path + f, 'r').readlines()
            
            with open(labelpath + fname+".txt", 'w') as fwriter:
                for line in content:
                    line = line.split(' ') # x1 y1 x2 y2 
                    if len(line) > 0:
                        x1 = float(line[0])
                        y1 = float(line[1])
                        x2 = float(line[2])
                        y2 = float(line[3])
                        cx = (x1 + x2) / 2. /256
                        cy = (y1 + y2) / 2. /256
                        w = (x2 - x1) /256
                        h = (y2 - y1) /256
                        fwriter.write("0 "+str(cx)+" "+str(cy)+" "+str(w)+" "+str(h)+"\n")

move_images()
gen_labels()
        


