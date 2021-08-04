import os
path = "/home/krf/datasets/clipart/labels/train/"

for lbl in os.listdir(path):
    f = open(path+lbl, 'w')
    f.close()