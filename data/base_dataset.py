"""这个模块实现了一个抽象基类（ABC）'BaseDataset' 用于数据集。

它还包括一些常见的变换函数（例如，get_transform, __scale_width），这些函数可以在子类中稍后使用。
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

# 定义一个抽象基类BaseDataset，继承自data.Dataset和ABC


class BaseDataset(data.Dataset, ABC):
    """这个类是一个抽象基类（ABC），用于数据集。

    要创建一个子类，你需要实现以下四个函数：
    -- <__init__>:                     初始化类，首先调用BaseDataset.__init__(self, opt).
    -- <__len__>:                      返回数据集的大小。
    -- <__getitem__>:                  获取一个数据点。
    -- <modify_commandline_options>:   (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """初始化类；将选项保存在类中

        参数:
            opt (Option class)-- 存储所有实验标志；需要是BaseOptions的子类
        """
        # 保存选项
        self.opt = opt
        # 保存数据根目录
        self.root = opt.dataroot

    @staticmethod
    # 修改命令行选项
    def modify_commandline_options(parser, is_train):
        """添加新的数据集特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否为训练阶段或测试阶段。你可以使用这个标志来添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。
        """
        # 返回修改后的解析器
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集中图像的总数。"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """返回一个数据点和其元数据信息。

        参数:
            index - - a random integer for data indexing

        Returns:
            一个包含数据及其名称的字典。它通常包含数据本身及其元数据信息。
        """
        pass


def get_params(opt, size):
    """获取图像的裁剪位置和翻转参数。

    参数:
        opt - - 选项类，包含训练参数
        size - - 图像的原始尺寸

    返回:
        {'crop_pos': (x, y), 'flip': flip} - - 包含裁剪位置和翻转参数的字典
    """
    w, h = size
    new_h = h
    new_w = w
    # 如果预处理方式为resize_and_crop，则将图像尺寸调整为指定尺寸
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    # 如果预处理方式为scale_width_and_crop，则将图像宽度调整为指定宽度，并计算新的高度
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    # 在新的图像尺寸范围内随机生成裁剪位置
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    # 随机生成一个0到1之间的浮点数，如果大于0.5，则进行翻转
    flip = random.random() > 0.5
    # 返回包含裁剪位置和翻转参数的字典
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """获取图像的变换列表。

    参数:
        opt - - 选项类，包含训练参数
        params - - 图像的裁剪位置和翻转参数
        grayscale - - 是否将图像转换为灰度
        method - - 图像缩放方法
        convert - - 是否将图像转换为张量
    """
    # 创建一个空的变换列表
    transform_list = []
    # 如果图像为灰度，则添加灰度变换
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # 如果预处理方式包含resize，则将图像调整为指定尺寸
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    # 如果预处理方式包含scale_width，则将图像宽度调整为指定宽度
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size, method)))

    # 如果预处理方式包含crop，则进行裁剪
    if 'crop' in opt.preprocess:
        # 如果params为None，则使用随机裁剪
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        # 否则使用自定义的裁剪函数
        else:
            transform_list.append(transforms.Lambda(
                lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    # 如果预处理方式为none，则将图像尺寸调整为2的幂次
    if opt.preprocess == 'none':
        # 将图像尺寸调整为2的幂次
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base=4, method=method)))

    # 如果预处理方式为none，则将图像尺寸调整为2的幂次
    if not opt.no_flip:
        # 如果params为None，则使用随机水平翻转
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        # 否则使用自定义的翻转函数
        elif params['flip']:
            # 使用自定义的翻转函数
            transform_list.append(transforms.Lambda(
                lambda img: __flip(img, params['flip'])))

    # 如果convert为True，则将图像转换为张量
    if convert:
        transform_list += [transforms.ToTensor()]
        # 如果图像为灰度，则进行归一化
        if grayscale:
            # 使用自定义的归一化函数
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # 返回一个Compose对象，包含所有变换
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    """将图像尺寸调整为2的幂次。

    参数:
        img - - 输入图像
        base - - 调整后的尺寸的基数
        method - - 图像缩放方法
    """
    # 获取图像的原始尺寸
    ow, oh = img.size
    # 将图像高度调整为2的幂次
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    # 打印图像尺寸警告信息（仅打印一次）
    __print_size_warning(ow, oh, w, h)
    # 将图像尺寸调整为指定尺寸
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    """将图像宽度调整为指定宽度。

    参数:
        img - - 输入图像
        target_width - - 目标宽度
        method - - 图像缩放方法
    """
    ow, oh = img.size
    # 如果图像宽度已经等于目标宽度，则直接返回图像
    if (ow == target_width):
        return img
    w = target_width
    # 计算新的高度
    h = int(target_width * oh / ow)
    # 将图像宽度调整为指定宽度
    return img.resize((w, h), method)


def __crop(img, pos, size):
    """裁剪图像。

    参数:
        img - - 输入图像
        pos - - 裁剪位置
        size - - 裁剪尺寸
    """
    # 获取图像的原始尺寸
    ow, oh = img.size
    # 获取裁剪位置
    x1, y1 = pos
    # 设置裁剪尺寸
    tw = th = size
    # 如果图像尺寸大于裁剪尺寸，则进行裁剪
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    """翻转图像。

    参数:
        img - - 输入图像
        flip - - 是否翻转
    """
    # 如果翻转标志为True，则进行水平翻转
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    # 否则返回原始图像
    return img


def __print_size_warning(ow, oh, w, h):
    """打印图像尺寸警告信息（仅打印一次）"""
    # 如果当前函数没有打印过，则打印警告信息
    if not hasattr(__print_size_warning, 'has_printed'):
        # 打印警告信息
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        # 设置当前函数已经打印过
        __print_size_warning.has_printed = True
