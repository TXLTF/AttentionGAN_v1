"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

# 将Tensor数组转换为numpy图像数组


def tensor2im(input_image, imtype=np.uint8):
    """将Tensor数组转换为numpy图像数组

    参数:
        input_image (tensor) -- 输入图像张量数组
        imtype (type)        -- 转换后的numpy数组类型
    """
    # 如果输入不是numpy数组
    if not isinstance(input_image, np.ndarray):
        # 如果输入是torch.Tensor
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            # 获取数据
            image_tensor = input_image.data
        else:
            return input_image
        # 将tensor转换为numpy数组
        image_numpy = image_tensor[0].cpu().float().numpy()
        # 如果输入是灰度图像，将其转换为RGB图像
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # 将灰度图像复制3次，以匹配RGB图像的维度
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # 将numpy数组进行转置，并进行归一化处理
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / \
            2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        # 如果输入是numpy数组，直接使用
        image_numpy = input_image
    # 将numpy数组转换为指定类型
    return image_numpy.astype(imtype)

# 计算并打印平均绝对梯度


def diagnose_network(net, name='network'):
    """计算并打印平均绝对梯度

    参数:
        net (torch network) -- Torch网络
        name (str) -- 网络名称
    """
    # 初始化平均值和计数器
    mean = 0.0
    count = 0
    # 遍历网络的参数
    for param in net.parameters():
        # 如果参数的梯度不为空
        if param.grad is not None:
            # 累加平均值
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    # 如果计数器大于0
    if count > 0:
        # 计算平均值
        mean = mean / count
    # 打印网络名称和平均值
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """将numpy图像保存到磁盘

    参数:
        image_numpy (numpy array) -- 输入numpy数组
        image_path (str)          -- 图像路径
    """
    # 将numpy数组转换为PIL图像
    image_pil = Image.fromarray(image_numpy)
    # 保存图像到指定路径
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """打印numpy数组的平均值、最小值、最大值、中位数、标准差和大小

    参数:
        val (bool) -- 如果打印numpy数组的值
        shp (bool) -- 如果打印numpy数组的形状
    """
    # 将numpy数组转换为float64类型
    x = x.astype(np.float64)
    # 如果打印numpy数组的形状
    if shp:
        print('shape,', x.shape)
    # 如果打印numpy数组的值
    if val:
        # 将numpy数组展平
        x = x.flatten()
        # 打印numpy数组的平均值、最小值、最大值、中位数和标准差
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """创建空目录，如果不存在

    参数:
        paths (str list) -- 目录路径列表
    """

    # 如果paths是列表且不是字符串
    if isinstance(paths, list) and not isinstance(paths, str):
        # 遍历目录路径列表
        for path in paths:
            # 创建目录
            mkdir(path)
    else:
        # 创建目录
        mkdir(paths)


def mkdir(path):
    """创建单个空目录，如果不存在

    参数:
        path (str) -- 单个目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
