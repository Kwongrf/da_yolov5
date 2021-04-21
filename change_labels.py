import os
from tqdm import tqdm
stage = ["train", "test", "val"]

path = "/home/krf/datasets/DA/foggy_cityscapes/labels/"
new_label = {10:1,11:0,20:3,21:2,24:4,26:5,27:6,31:7}# foggy
# new_label = {8:0 ,9:1,18:2,19:3,20:4,23:5,26:6,30:7}#{4:0, 10:1, 14:2, 20:3, 21:4, 26:5, 27:6, 32:7}
for s in stage:
    for lbl in tqdm(os.listdir(path + s)):
        f = open(path + s+"/"+ lbl, 'r')
        boxes = f.readlines()
        new_boxes = []
        for box in boxes:
            new_box = []
            box = box.strip().split(' ')
            cls = int(box[0])
            if cls not in new_label:
                continue
            cls = new_label[cls]

            new_box.append(cls)
            for i in range(1, len(box)):
                new_box.append(float(box[i]))
            new_boxes.append(new_box)
        f.close()
        f = open(path + s+"/"+ lbl, 'w')
        for box in new_boxes:
            print(box)
            f.write(str(box)[1:-1].replace(',','')+"\n")
        f.close()


            

