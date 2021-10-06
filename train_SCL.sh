#python train_SCL.py --data cityscape2foggy.yaml --cfg yolov5s.yaml --weights "yolov5s.pt" --batch-size 32 --device 0,1 --img-size 640 --epochs 100 --name scl_v5s100e
# python train_SCL.py --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt" --batch-size 8 --device 0,1 --img-size 640 --epochs 300 --name scl_v5m300e
# python train_SCL.py --hyp hyp.finetune.yaml --data voc2clipart.yaml --cfg yolov5m.yaml --weights "runs/train/v5m100e_6disc_fullvoc7/weights/best.pt" --batch-size 64 --device 0,1 --img-size 320 --epochs 100 --name v5s100e_6_fullvoc_finetune --cache-images --da 6 --workers 4 --resume "runs/train/v5s100e_6_fullvoc_finetune2/weights/last.pt"
# python train_SCL.py --data voc2clipart.yaml --cfg yolov5s.yaml --weights "yolov5s.pt" --batch-size 96 --device 0,1 --img-size 320 --epochs 100 --name v5s100e_3_fullvoc --cache-images --da 3
# python train_SCL.py --data voc2clipart.yaml --cfg yolov5s.yaml --weights "yolov5s.pt" --batch-size 96 --device 0,1 --img-size 320 --epochs 100 --name v5s100e_4_fullvoc --cache-images --da 4
# python train_SCL.py --hyp hyp.scatch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "runs/train/scl_v5m300e4/weights/best.pt"  --batch-size 8 --device 2 --img-size 640 --epochs 300 --name inst_align_v5m100e --da 9

# python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 2 --img-size 640 --epochs 300 --name inst_align_v5m300e_8da_instw --da 8 --cache-images
# python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 3 --img-size 640 --epochs 300 --name inst_align_v5m300e_6da_headw --da 6 --cache-images

# python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 3  --img-size 640 --epochs 300 --name inst_align_v5m300e_6da_nopre --da 6 --resume "runs/train/inst_align_v5m300e_6da_nopre/weights/last.pt" 
# python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 2 --img-size 640 --epochs 300 --name inst_align_v5m300e_7da_nopre --da 7
# python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 2 --img-size 640 --epochs 300 --name inst_align_v5m300e_7da_nopre --da 8

python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 2  --img-size 640 --epochs 300 --name inst_align_v5m300e_8da_0.3cls_1.0obj --da 8 --cache-images
python train_SCL.py --hyp hyp.finetune.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "runs/train/inst_align_v5m300e_4da_nopre/weights/best.pt"  --batch-size 8 --device 3 --img-size 640 --epochs 100 --name inst_align_v5m100e_4da_finetune --da 4 --cache-images --resume "runs/train/inst_align_v5m100e_4da_finetune2/weights/last.pt"
python train_SCL.py --hyp hyp.scratch.yaml  --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights "yolov5m.pt"  --batch-size 8 --device 1 --img-size 640 --epochs 300 --name v5m300e_4da_less_clsobjloss --da 4 --cache-images 
python train_SCL_3stream.py --hyp hyp.scratch.yaml --data cityscape2foggy.yaml --cfg yolov5m.yaml --weights yolov5m.pt --batch-size 16 --device 3 --img-size 640 --epochs 300 --name v5m300e_3stream_pl --da 8
