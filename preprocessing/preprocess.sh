python dataset_tools.py convert --source=/mnt/SSD/lengx/Data/imagenet_series/imagenet/train --dest=../data/images.zip --resolution=256x256 --transform=center-crop-dhariwal
python dataset_tools.py encode --source=../data/images.zip --dest=../data/vae-sd.zip