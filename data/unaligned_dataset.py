import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

# 定义一个继承自BaseDataset的UnalignedDataset类


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    它需要两个目录来托管来自域A的训练图像 '/path/to/data/trainA'
    和来自域B的图像 '/path/to/data/trainB' 分别。
    你可以使用数据集标志 '--dataroot /path/to/data' 训练模型。
    同样，你需要在测试时准备两个目录：
    '/path/to/data/testA' and '/path/to/data/testB' 在测试时。
    """

    # 初始化方法，接受一个opt参数，用于存储实验标志
    def __init__(self, opt):
        """
        初始化这个数据集类。

        参数:
            opt (Option class) -- 存储所有实验标志；需要是BaseOptions的子类
        """

        # 调用父类BaseDataset的初始化方法，传入opt参数
        BaseDataset.__init__(self, opt)
        # 创建路径 '/path/to/data/trainA'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # 创建路径 '/path/to/data/trainB'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        # load images from '/path/to/data/trainA'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        # load images from '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        # 获取数据集A的大小
        self.A_size = len(self.A_paths)  # get the size of dataset A
        # 获取数据集B的大小
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # 根据方向判断是否是BtoA
        btoA = self.opt.direction == 'BtoA'
        # get the number of channels of input image
        # 根据方向判断输入通道数
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        # get the number of channels of output image
        # 根据方向判断输出通道数
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        # 获取变换函数
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """
        返回一个数据点和它的元数据信息。

        参数:
            index (int)      -- 一个随机整数用于数据索引

        返回一个字典，包含A, B, A_paths和B_paths
            A (tensor)       -- 输入域中的图像
            B (tensor)       -- 目标域中的对应图像
            A_paths (str)    -- 图像路径
            B_paths (str)    -- 图像路径
        """
        # 根据方向判断是否是BtoA
        A_path = self.A_paths[index %
                              self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """
        返回数据集中图像的总数。

        由于我们有两个数据集，可能具有不同数量的图像，
        我们取最大值
        """
        return max(self.A_size, self.B_size)
