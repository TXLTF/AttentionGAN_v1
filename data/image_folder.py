"""一个修改后的图像文件夹类

我们修改了官方的PyTorch图像文件夹（https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py）
使得这个类可以加载当前目录和其子目录中的图像。
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

# 支持的图像文件扩展名
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    """检查文件名是否以IMG_EXTENSIONS中的任意一个扩展名结尾"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """生成一个包含图像路径的列表"""
    # 初始化一个空列表，用于存储图像路径
    images = []
    # 检查输入的目录是否存在
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    # 遍历目录中的所有文件和子目录
    for root, _, fnames in sorted(os.walk(dir)):
        # 遍历当前目录中的所有文件名
        for fname in fnames:
            # 检查文件名是否以IMG_EXTENSIONS中的任意一个扩展名结尾
            if is_image_file(fname):
                # 将文件路径添加到图像路径列表中
                path = os.path.join(root, fname)
                images.append(path)
    # 返回图像路径列表的前max_dataset_size个元素
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    """默认的图像加载函数"""
    # 打开图像文件并将其转换为RGB格式
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        """初始化ImageFolder类"""
        # 生成图像路径列表
        imgs = make_dataset(root)
        # 如果图像路径列表为空，则抛出错误
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        """获取指定索引处的图像数据"""
        path = self.imgs[index]
        # 使用加载器函数加载图像
        img = self.loader(path)
        # 如果存在变换函数，则应用变换
        if self.transform is not None:
            img = self.transform(img)
        # 如果需要返回路径，则返回图像和路径
        if self.return_paths:
            return img, path
        # 否则只返回图像
        return img

    def __len__(self):
        """返回图像数据集的长度"""
        return len(self.imgs)
