#python train_SCL.py --data cityscape2foggy.yaml --cfg yolov5s.yaml --weights "yolov5s.pt" --batch-size 32 --device 0,1 --img-size 640 --epochs 100 --name scl_v5s100e
# python train_SCL.py --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt" --batch-size 8 --device 0,1 --img-size 640 --epochs 300 --name scl_v5m300e
python train_SCL.py --data voc2clipart.yaml --cfg yolov5s.yaml --weights "yolov5s.pt" --batch-size 24 --device 0,3 --img-size 300 --epochs 300 --name scl_v5s300e


