import random
import torch


class ImagePool():
    """这个类实现了一个图像缓冲区，该缓冲区存储先前生成的图像。

    这个缓冲区使我们能够使用以前生成的图像更新鉴别器，而不是使用最新的生成器生成的图像。
    """
# 初始化图像池

    def __init__(self, pool_size):
        """初始化ImagePool类

        Parameters:
            pool_size (int) -- 图像缓冲区的大小，如果pool_size=0，则不创建缓冲区
        """
        # 初始化图像池的大小
        self.pool_size = pool_size
        # 如果池的大小大于0，则创建一个空池
        if self.pool_size > 0:  # create an empty pool
            # 初始化图像数量为0
            self.num_imgs = 0
            # 创建一个空列表来存储图像
            self.images = []

    def query(self, images):
        """从缓冲区返回一张图像。

        Parameters:
            images: 生成器生成的最新图像

        返回缓冲区中的图像。

        50/100，缓冲区将返回输入图像。
        50/100，缓冲区将返回先前存储在缓冲区中的图像，
        并将当前图像插入缓冲区。
        """

        if self.pool_size == 0:  # 如果缓冲区大小为0，什么都不做
            return images
        # 创建一个空列表来存储返回的图像
        return_images = []
        # 遍历输入的图像
        for image in images:
            # 将图像转换为张量，并添加一个维度
            image = torch.unsqueeze(image.data, 0)
            # 如果缓冲区未满，则将当前图像插入缓冲区
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # 以50%的概率返回先前存储的图像，并将当前图像插入缓冲区
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1)  # randint is inclusive
                    # 从缓冲区中随机选择一个图像
                    tmp = self.images[random_id].clone()
                    # 将当前图像插入缓冲区
                    self.images[random_id] = image
                    # 将先前存储的图像添加到返回的图像列表中
                    return_images.append(tmp)
                # 以50%的概率返回当前图像
                else:       # by another 50% chance, the buffer will return the current image
                    # 将当前图像添加到返回的图像列表中
                    return_images.append(image)
        # collect all the images and return
        # 将返回的图像列表转换为张量，并添加一个维度
        return_images = torch.cat(return_images, 0)
        return return_images
