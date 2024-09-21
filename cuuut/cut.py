import cv2
import os

# 设置裁剪的宽度和高度
crop_w = 128
crop_h = 128

for filename in os.listdir('../datasets/TESTDATASET4Perfect_cutbefore/testA'):
    image = cv2.imread(os.path.join('../datasets/TESTDATASET4Perfect/testA/', filename))
    height, width, _ = image.shape

    # 创建保存小图片的目录
    save_dir = '../datasets/TESTDATASET4Perfect_cut/testA'
    os.makedirs(save_dir, exist_ok=True)

    # 裁剪图片
    for y in range(0, height, crop_h):
        for x in range(0, width, crop_w):
            crop = image[y:y + crop_h, x:x + crop_w]
            cv2.imwrite(os.path.join(save_dir, f'crop_{filename}_{y}_{x}.jpg'), crop)