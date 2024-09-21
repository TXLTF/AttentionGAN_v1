import model  # 假设这是你的模型库

# 处理每个裁剪的小图片
for filename in os.listdir(save_dir):
    img_path = os.path.join(save_dir, filename)
    img = cv2.imread(img_path)

    # 经过模型生成
    generated_img = model.generate(img)  # 假设有一个 generate 函数

    # 保存生成的图片
    cv2.imwrite(os.path.join('./generated_images/', filename), generated_img)