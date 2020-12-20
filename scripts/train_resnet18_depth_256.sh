TRAIN_SET=../../Dataset/kitti_256
python ../train.py $TRAIN_SET \
--num-scales 1 --epochs 50 \
-b6 -ds 0.1 -dc 0.5 -cc 0 --epoch-size 0 --sequence-length 3 \
--with-mask 1 \
--with-triangulation 1 \
--pretrained-disp ../pretrained/dispnet_initial.pth.tar \
--pretrained-flow ../pretrained/kitti_flow.pth \
--with-gt \
--log-output \
--name DispResNet_triangulation