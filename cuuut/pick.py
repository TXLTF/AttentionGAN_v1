import numpy as np
import os
import cv2

# 设置裁剪的宽度和高度
crop_w = 128
crop_h = 128

height = 128
width = 128
# 获取生成图片的尺寸
generated_images = []
# for filename in os.listdir('./testC/'):
#     img = cv2.imread(os.path.join('./testC/', filename))
for filename in os.listdir('../datasets/TESTDATASET4Perfect_cutbefore/testA/'):
    img = cv2.imread(os.path.join('../datasets/TESTDATASET4Perfect_cutbefore/testA/', filename))
    generated_images.append(img)

# 假设生成的图片尺寸与裁剪尺寸一致
# rows = height // crop_h
# cols = width // crop_w
rows = 4
cols = 4

# 拼接图片
big_image = np.zeros((height, width, 3), dtype=np.uint8)
for idx, img in enumerate(generated_images):
    y = (idx // cols) * crop_h
    x = (idx % cols) * crop_w
    big_image[y:y+crop_h, x:x+crop_w] = img

# 保存拼接后的大图片
cv2.imwrite('reconstructed_image.jpg', big_image)