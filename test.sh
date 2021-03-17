python test.py --weight 'runs/train/noda/weights/last.pt' --data bdd_day.yaml --batch-size 64 --device 3 --img-size 256 --task test

# model           P           R           mAP@.5      mAP@.5:.95
# no_da_500e      0.825       0.728       0.817       0.501
# da_300e         0.79        0.726       0.789       0.478


# pretrained_da   0.865        0.76       0.857       0.559
# no_da           0.836        0.783      0.861       0.549

# pretrain_disc   0.815        0.77       0.842       0.536 
# 3_DA            0.836        0.765      0.836       0.532

