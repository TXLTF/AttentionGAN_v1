import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


###############################################################################
# Helper Functions
###############################################################################

# 定义一个Identity类，继承自nn.Module
class Identity(nn.Module):
    """
    一个简单的Identity类，用于在不需要任何操作的情况下直接返回输入。
    """

    def forward(self, x):
        return x

# 定义一个函数，用于根据给定的norm_type返回一个归一化层


def get_norm_layer(norm_type='instance'):
    """返回一个归一化层

    参数:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    # 如果norm_type是batch，则使用BatchNorm2d
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)
    # 如果norm_type是instance，则使用InstanceNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # 如果norm_type是none，则使用Identity
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        # 如果norm_type不是以上三种，则抛出NotImplementedError
        # 在这种情况下，函数会抛出一个NotImplementedError异常。
        # 异常消息会包含未识别的norm_type值，以便于调试。
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer

# 定义一个函数，用于根据给定的optimizer和opt返回一个学习率调度器


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    参数:
        optimizer -- 网络的优化器
        opt (option class) -- 存储所有实验标志；需要是BaseOptions的子类
                              opt.lr_policy是学习率策略的名称：linear | step | plateau | cosine

    对于'linear'，我们保持相同的学习率，直到<opt.niter> epochs，然后线性衰减到零，直到<opt.niter_decay> epochs。
    对于其他调度器（step、plateau和cosine），我们使用默认的PyTorch调度器。
    查看https://pytorch.org/docs/stable/optim.html了解更多细节。
    """
    # 如果opt.lr_policy是linear，则使用线性衰减的学习率调度器
    if opt.lr_policy == 'linear':
        # 定义一个lambda_rule函数，用于计算当前epoch的学习率
        def lambda_rule(epoch):
            # 计算当前epoch的学习率
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        # 使用LambdaLR调度器，根据lambda_rule函数计算学习率
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # 如果opt.lr_policy是step，则使用StepLR
    elif opt.lr_policy == 'step':
        # 使用StepLR调度器，根据opt.lr_decay_iters计算学习率
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # 如果opt.lr_policy是plateau，则使用ReduceLROnPlateau
    elif opt.lr_policy == 'plateau':
        # 使用ReduceLROnPlateau调度器，根据mode、factor、threshold和patience计算学习率
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # 如果opt.lr_policy是cosine，则使用CosineAnnealingLR
    elif opt.lr_policy == 'cosine':
        # 使用CosineAnnealingLR调度器，根据opt.niter计算学习率
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# 定义一个函数，用于初始化网络权重


def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重。

    参数:
        net (network)   -- 要初始化的网络
        init_type (str) -- 初始化方法的名称：normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 用于normal、xavier和orthogonal的缩放因子。

    我们使用'normal'在原始的pix2pix和CycleGAN论文中。但是xavier和kaiming可能对某些应用更好。
    请随意尝试自己。
    """
    # 定义一个init_func函数，用于初始化网络权重
    def init_func(m):  # define the initialization function
        # 获取当前模块的类名
        classname = m.__class__.__name__
        # 如果当前模块有weight属性，并且类名包含Conv或Linear，则进行初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 根据init_type选择初始化方法
            if init_type == 'normal':
                # 使用init.normal_方法对权重进行正态分布初始化
                init.normal_(m.weight.data, 0.0, init_gain)
            # 如果init_type是xavier，则使用Xavier正态分布初始化
            elif init_type == 'xavier':
                # 使用init.xavier_normal_方法对权重进行Xavier正态分布初始化
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # 使用init.kaiming_normal_方法对权重进行Kaiming正态分布初始化
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            # 如果init_type是orthogonal，则使用正交初始化
            elif init_type == 'orthogonal':
                # 使用init.orthogonal_方法对权重进行正交初始化
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                # 如果init_type不是以上四种，则抛出NotImplementedError
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            # 如果当前模块有bias属性，则进行初始化
            if hasattr(m, 'bias') and m.bias is not None:
                # 使用init.constant_方法将bias初始化为0
                init.constant_(m.bias.data, 0.0)
        # 如果当前模块是BatchNorm2d，则进行初始化
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            # 使用init.normal_方法对权重进行正态分布初始化
            init.normal_(m.weight.data, 1.0, init_gain)
            # 使用init.constant_方法将bias初始化为0
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    # 使用apply方法将init_func应用到网络的每个模块
    net.apply(init_func)  # apply the initialization function <init_func>

# 定义一个函数，用于初始化网络


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化一个网络：1. 注册CPU/GPU设备（支持多GPU）；2. 初始化网络权重
    参数:
        net (网络) -- 要初始化的网络
        init_type (str)    -- 初始化方法的名称：normal | xavier | kaiming | orthogonal
        gain (float)       -- 用于normal、xavier和orthogonal的缩放因子。
        gpu_ids (int list) -- 网络运行的GPU：例如，0,1,2

    返回一个初始化的网络。
    """
    # 如果gpu_ids列表不为空，则将网络移动到指定的GPU
    if len(gpu_ids) > 0:
        # 确保GPU可用
        assert (torch.cuda.is_available())
        # print(net)
        net.to(gpu_ids[0])
        # 将网络包装为DataParallel对象，以便在多个GPU上并行运行
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # 调用init_weights函数初始化网络权重
    init_weights(net, init_type, init_gain=init_gain)
    return net

# 定义一个函数，用于根据输入的参数创建一个生成器


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """创建一个生成器

    参数:
        input_nc (int) -- 输入图像的通道数
        output_nc (int) -- 输出图像的通道数
        ngf (int) -- 最后一个卷积层的滤波器数量
        netG (str) -- 架构的名称：resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- 网络中使用的归一化层的名称：batch | instance | none
        use_dropout (bool) -- 如果使用dropout层。
        init_type (str)    -- 我们初始化方法的名称。
        init_gain (float)  -- 用于normal、xavier和orthogonal的缩放因子。
        gpu_ids (int list) -- 网络运行的GPU：例如，0,1,2

    Returns a generator

    我们的当前实现提供了两种类型的生成器：
        U-Net: [unet_128] (for 128x128输入图像) 和 [unet_256] (for 256x256输入图像)
        原始U-Net论文：https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        我们改编了Justin Johnson的神经风格转移项目（https://github.com/jcjohnson/fast-neural-style）的Torch代码。

    生成器已通过<init_net>初始化。它使用RELU作为非线性。
    """
    # 初始化一个空的生成器
    net = None
    # 根据norm参数获取归一化层
    norm_layer = get_norm_layer(norm_type=norm)

    # 根据netG参数选择生成器架构
    if netG == 'resnet_9blocks':
        # 创建一个ResnetGenerator对象，并使用init_net函数初始化
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        # 创建一个ResnetGenerator对象，并使用init_net函数初始化
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        # 创建一个UnetGenerator对象，并使用init_net函数初始化
        net = UnetGenerator(input_nc, output_nc, 7, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        # 创建一个UnetGenerator对象，并使用init_net函数初始化
        net = UnetGenerator(input_nc, output_nc, 8, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'our':
        # 创建一个ResnetGenerator_our对象，并使用init_net函数初始化
        net = ResnetGenerator_our(input_nc, output_nc, ngf, n_blocks=9)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

# 定义一个函数，用于根据输入的参数创建一个判别器


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- 输入图像的通道数
        ndf (int)          -- 第一个卷积层的滤波器数量
        netD (str)         -- 架构的名称：basic | n_layers | pixel
        n_layers_D (int)   -- 判别器中卷积层的数量；当netD=='n_layers'时有效
        norm (str)         -- 网络中使用的归一化层的名称：batch | instance | none
        init_type (str)    -- 初始化方法的名称：normal | xavier | kaiming | orthogonal
        init_gain (float)  -- 用于normal、xavier和orthogonal的缩放因子。
        gpu_ids (int list) -- 网络运行的GPU：例如，0,1,2

    返回一个判别器

    我们的当前实现提供了三种类型的判别器：
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        它可以将70×70重叠的补丁分类为真实或虚假。
        这种补丁级别的判别器架构比全图像判别器具有更少的参数，并且可以以完全卷积的方式处理任意大小的图像。

        [n_layers]: 使用此模式，您可以通过参数<n_layers_D>指定判别器中的卷积层数量（默认值为3，如[basic]（PatchGAN）中所用）。

        [pixel]: 1x1 PixelGAN判别器可以判断一个像素是否为真实或虚假。
        它鼓励更大的颜色多样性，但对空间统计没有影响。

        判别器已通过<init_net>初始化。它使用Leakly RELU作为非线性。
        它可以判断70×70重叠的补丁是否为真实或虚假。
        这种补丁级别的判别器架构比全图像判别器具有更少的参数，并且可以以完全卷积的方式处理任意大小的图像。
        in a fully convolutional fashion.

        [n_layers]: 使用此模式，您可以通过参数<n_layers_D>指定判别器中的卷积层数量（默认值为3，如[basic]（PatchGAN）中所用）。

        [pixel]: 1x1 PixelGAN判别器可以判断一个像素是否为真实或虚假。
        它鼓励更大的颜色多样性，但对空间统计没有影响。

    判别器已通过<init_net>初始化。它使用Leakly RELU作为非线性。
    """
    # 初始化一个空的判别器
    net = None
    # 根据norm参数获取归一化层
    norm_layer = get_norm_layer(norm_type=norm)

    # 根据netD参数选择判别器架构
    if netD == 'basic':  # default PatchGAN classifier
        # 创建一个NLayerDiscriminator对象，并使用init_net函数初始化
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        # 创建一个NLayerDiscriminator对象，并使用init_net函数初始化
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        # 创建一个PixelDiscriminator对象，并使用init_net函数初始化
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


# 定义不同类型的GAN损失函数
##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """定义不同的GAN目标。

    GANLoss类抽象了创建与输入相同大小的标签张量的需要
    该类抽象了创建与输入相同大小的标签张量的需要。
    """

    # 初始化GANLoss类
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """初始化GANLoss类。

        参数:
            gan_mode (str) - - GAN目标的类型。它目前支持vanilla、lsgan和wgangp。
            target_real_label (bool) - - 真实图像的标签
            target_fake_label (bool) - - 虚假图像的标签

        注意：不要在判别器最后一层使用sigmoid。
        LSGAN需要没有sigmoid。vanilla GANs将使用BCEWithLogitsLoss处理它。
        """
        # 调用父类nn.Module的初始化方法
        super(GANLoss, self).__init__()
        # 注册一个缓冲区，用于存储真实标签
        self.register_buffer('real_label', torch.tensor(target_real_label))
        # 注册一个缓冲区，用于存储虚假标签
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        # 将gan_mode的值存储在类中
        self.gan_mode = gan_mode
        # 根据gan_mode的值选择损失函数
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        # 如果gan_mode是lsgan，则使用均方误差损失函数
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        # 如果gan_mode是vanilla，则使用二元交叉熵损失函数
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    # 定义一个方法，用于创建与输入相同大小的标签张量
    def get_target_tensor(self, prediction, target_is_real):
        """创建与输入相同大小的标签张量。

        参数:
            prediction (tensor) - - 通常是判别器的输出
            target_is_real (bool) - - 如果ground truth标签是真实图像还是虚假图像

        返回:
            A label tensor filled with ground truth label, and with the size of the input
        """
        # 如果target_is_real为真，则将真实标签存储在target_tensor中
        if target_is_real:
            target_tensor = self.real_label
        else:
            # 否则，将虚假标签存储在target_tensor中
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    # 定义一个方法，用于计算给定判别器输出和真实标签的损失
    def __call__(self, prediction, target_is_real):
        """计算给定判别器输出和真实标签的损失。

        参数:
            prediction (tensor) - - 通常是判别器的输出
            target_is_real (bool) - - 如果ground truth标签是真实图像还是虚假图像

        返回:
            计算的损失。
        """
        # 如果gan_mode是lsgan或vanilla，则计算损失
        if self.gan_mode in ['lsgan', 'vanilla']:
            # 调用get_target_tensor方法创建与输入相同大小的标签张量
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # 使用损失函数计算损失
            loss = self.loss(prediction, target_tensor)
        # 如果gan_mode是wgangp，则计算损失
        elif self.gan_mode == 'wgangp':
            # 如果target_is_real为真，则计算损失
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

# 定义一个函数，用于计算梯度惩罚损失


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """计算梯度惩罚损失，用于WGAN-GP论文https://arxiv.org/abs/1704.00028

    参数:
        netD (网络) -- 判别器网络
        real_data (tensor array) -- 真实图像
        fake_data (tensor array) -- 生成器生成的图像
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- 如果我们混合真实数据和虚假数据，则为[real | fake | mixed]。
        constant (float)            -- 公式中使用的常数( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- 这个损失的权重

    返回梯度惩罚损失
    """

    # 如果lambda_gp大于0.0，则计算梯度惩罚损失
    if lambda_gp > 0.0:
        # 根据type参数选择真实数据、虚假数据或两者的线性插值
        # either use real images, fake images, or a linear interpolation of two.
        if type == 'real':
            # 如果type是real，则使用真实数据
            interpolatesv = real_data
        elif type == 'fake':
            # 如果type是fake，则使用虚假数据
            interpolatesv = fake_data
            # 如果type是mixed，则使用真实数据和虚假数据的线性插值
        elif type == 'mixed':
            # 生成一个与真实数据形状相同的随机数
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            # 将随机数扩展为与真实数据形状相同的张量
            alpha = alpha.expand(real_data.shape[0], real_data.nelement(
            ) // real_data.shape[0]).contiguous().view(*real_data.shape)
            # 计算真实数据和虚假数据的线性插值
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        # 将interpolatesv的requires_grad属性设置为True，表示需要计算梯度
        interpolatesv.requires_grad_(True)
        # 将interpolatesv输入到判别器netD中，得到输出disc_interpolates
        disc_interpolates = netD(interpolatesv)
        # 计算interpolatesv的梯度
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(
                                            disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        # 将gradients展平为一维张量
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        # 计算梯度惩罚损失
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) -
                            constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# 基于ResNet架构的生成器


class ResnetGenerator(nn.Module):
    """基于ResNet的生成器，由几个下采样/上采样操作之间的ResNet块组成。

    我们改编了PyTorch代码和Justin Johnson的神经风格转移项目(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """构造一个基于ResNet的生成器

        参数:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层的滤波器数量
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 如果使用dropout层
            n_blocks (int)      -- ResNet块的数量
            padding_type (str)  -- 卷积层的填充类型: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        # 如果norm_layer是nn.InstanceNorm2d，则设置use_bias为True，否则为False
        if type(norm_layer) == functools.partial:
            # 如果norm_layer是nn.InstanceNorm2d，则设置use_bias为True，否则为False
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 构建模型
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7,
                           padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        # 添加下采样层
        for i in range(n_downsampling):  # add downsampling layers
            # 计算当前层的滤波器数量
            mult = 2 ** i
            # 添加卷积层
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # 计算当前层的滤波器数量
        mult = 2 ** n_downsampling

        # 添加ResNet块
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # 添加上采样层
        for i in range(n_downsampling):  # add upsampling layers
            # 计算当前层的滤波器数量
            mult = 2 ** (n_downsampling - i)
            # 添加反卷积层
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # 添加反射填充层和卷积层
        model += [nn.ReflectionPad2d(3)]
        # 添加卷积层
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # 添加Tanh激活函数
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetGenerator_our(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        """构造一个基于ResNet的生成器

        参数:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层的滤波器数量
            n_blocks (int)      -- ResNet块的数量
        """

        super(ResnetGenerator_our, self).__init__()
        # 初始化输入和输出通道数
        self.input_nc = input_nc
        self.output_nc = output_nc
        # 初始化滤波器数量
        self.ngf = ngf
        # 初始化ResNet块的数量
        self.nb = n_blocks
        # 初始化第一个卷积层
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        # 初始化第一个归一化层
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)
        # 初始化ResNet块列表
        self.resnet_blocks = []
        # 遍历ResNet块的数量
        for i in range(n_blocks):
            # 创建ResNet块并添加到列表中
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))
            # 初始化权重
            self.resnet_blocks[i].weight_init(0, 0.02)
        # 将ResNet块列表转换为Sequential容器
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        # self.resnet_blocks1 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks1.weight_init(0, 0.02)
        # self.resnet_blocks2 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks2.weight_init(0, 0.02)
        # self.resnet_blocks3 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks3.weight_init(0, 0.02)
        # self.resnet_blocks4 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks4.weight_init(0, 0.02)
        # self.resnet_blocks5 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks5.weight_init(0, 0.02)
        # self.resnet_blocks6 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks6.weight_init(0, 0.02)
        # self.resnet_blocks7 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks7.weight_init(0, 0.02)
        # self.resnet_blocks8 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks8.weight_init(0, 0.02)
        # self.resnet_blocks9 = resnet_block(256, 3, 1, 1)
        # self.resnet_blocks9.weight_init(0, 0.02)

        # 添加反卷积层
        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        # 添加反卷积归一化层
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = nn.Conv2d(ngf, 27, 7, 1, 0)

        # 添加反卷积层
        self.deconv1_attention = nn.ConvTranspose2d(
            ngf * 4, ngf * 2, 3, 2, 1, 1)
        # 添加反卷积归一化层
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)

        # 添加Tanh激活函数
        self.tanh = torch.nn.Tanh()

    # 初始化权重
    def weight_init(self, mean, std):
        # 遍历所有模块并初始化权重
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # 输入图像经过卷积层和激活函数处理
    def forward(self, input):
        """前向传播"""
        # 使用反射填充
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        # 经过卷积层和激活函数处理
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        # 经过ResNet块处理
        x = self.resnet_blocks(x)
        # x = self.resnet_blocks1(x)
        # x = self.resnet_blocks2(x)
        # x = self.resnet_blocks3(x)
        # x = self.resnet_blocks4(x)
        # x = self.resnet_blocks5(x)
        # x = self.resnet_blocks6(x)
        # x = self.resnet_blocks7(x)
        # x = self.resnet_blocks8(x)
        # x = self.resnet_blocks9(x)
        # 经过反卷积层和激活函数处理
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        # 经过反卷积层和激活函数处理
        x_content = F.relu(self.deconv2_norm_content(
            self.deconv2_content(x_content)))
        # 使用反射填充
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        # 经过反卷积层处理
        content = self.deconv3_content(x_content)
        # 使用Tanh激活函数处理
        image = self.tanh(content)
        # 将图像分割为9个部分
        image1 = image[:, 0:3, :, :]
        # print(image1.size()) # [1, 3, 256, 256]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        # image10 = image[:, 27:30, :, :]

        # 经过反卷积层和激活函数处理
        x_attention = F.relu(
            self.deconv1_norm_attention(self.deconv1_attention(x)))
        # 经过反卷积层和激活函数处理
        x_attention = F.relu(self.deconv2_norm_attention(
            self.deconv2_attention(x_attention)))
        # 使用反射填充
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        # 经过反卷积层处理
        attention = self.deconv3_attention(x_attention)

        # 使用Softmax函数处理注意力权重
        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        # 将注意力权重分割为9个部分
        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        # 将注意力权重重复3次，以匹配图像的通道数
        attention1 = attention1_.repeat(1, 3, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        # 将图像与注意力权重相乘，以生成输出
        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        # output10 = image10 * attention10
        # 将输入图像与注意力权重相乘，以生成输出
        output10 = input * attention10

        # 将所有输出相加，以生成最终输出
        o = output1 + output2 + output3 + output4 + output5 + \
            output6 + output7 + output8 + output9 + output10

        return o, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10, attention1, attention2, attention3, attention4, attention5, attention6, attention7, attention8, attention9, attention10, image1, image2, image3, image4, image5, image6, image7, image8, image9

# resnet block with reflect padding


class resnet_block(nn.Module):
    """定义一个ResNet块"""

    def __init__(self, channel, kernel, stride, padding):
        """初始化"""
        super(resnet_block, self).__init__()
        # 初始化通道数、卷积核大小、步幅和填充
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        # 初始化第一个卷积层
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        # 初始化第一个归一化层
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    # weight_init
    # 初始化权重
    def weight_init(self, mean, std):
        # 遍历所有模块并初始化权重
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward
    # 前向传播
    def forward(self, input):
        # 使用反射填充
        x = F.pad(input, (self.padding, self.padding,
                  self.padding, self.padding), 'reflect')
        # 经过第一个卷积层和激活函数处理
        x = F.relu(self.conv1_norm(self.conv1(x)))
        # 使用反射填充
        x = F.pad(x, (self.padding, self.padding,
                  self.padding, self.padding), 'reflect')
        # 经过第二个卷积层和归一化处理
        x = self.conv2_norm(self.conv2(x))
        # 返回输入与处理结果的和
        return input + x

# 初始化权重


def normal_init(m, mean, std):
    """初始化权重"""
    # 如果m是卷积层或反卷积层
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # 使用正态分布初始化权重
        m.weight.data.normal_(mean, std)
        # 将偏置初始化为0
        m.bias.data.zero_()


# 定义Resnet块
class ResnetBlock(nn.Module):
    """定义一个ResNet块"""

    # 初始化
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化ResNet块

        A resnet block is a conv block with skip connections
        我们使用build_conv_block函数构建一个卷积块，
        并在<forward>函数中实现跳过连接。
        原始ResNet论文: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        # 构建卷积块
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    # 构建卷积块
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构建一个卷积块

        参数:
            dim (int)           -- 卷积层的通道数
            padding_type (str)  -- 填充类型: reflect | replicate | zero
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否使用dropout层
            use_bias (bool)     -- 卷积层是否使用偏置

        返回一个卷积块（包含一个卷积层、一个归一化层和一个非线性层（ReLU））
        """
        # 初始化卷积块
        conv_block = []
        # 初始化填充
        p = 0
        # 如果填充类型为反射
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        # 如果填充类型为复制
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        # 如果填充类型为零
        elif padding_type == 'zero':
            p = 1
        # 如果填充类型未实现
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        # 添加卷积层、归一化层和激活层
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        # 如果使用dropout
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # 初始化填充
        p = 0
        # 如果填充类型为反射
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        # 如果填充类型为复制
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        # 如果填充类型为零
            p = 1
        # 如果填充类型未实现
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        # 添加卷积层、归一化层和激活层
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=use_bias), norm_layer(dim)]

        # 返回卷积块
        return nn.Sequential(*conv_block)

    # 前向传播
    def forward(self, x):
        """Forward function (with skip connections)"""
        # 将输入与卷积块处理结果相加
        out = x + self.conv_block(x)  # add skip connections
        return out

# 定义Unet生成器


class UnetGenerator(nn.Module):
    """创建一个基于Unet的生成器"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构造一个Unet生成器
        参数:
            input_nc (int)  -- 输入图像的通道数
            output_nc (int) -- 输出图像的通道数
            num_downs (int) -- UNet的降采样次数。例如，如果|num_downs| == 7，
                              图像的大小将从128x128变为1x1 # 在瓶颈处
            ngf (int)       -- 最后一个卷积层的滤波器数量
            norm_layer      -- 归一化层

        我们从最内层到最外层构建U-Net。
        这是一个递归过程。
        """
        # 初始化
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # 初始化UnetSkipConnectionBlock
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        # add intermediate layers with ngf * 8 filters
        # 遍历num_downs - 5次
        for i in range(num_downs - 5):
            # 初始化UnetSkipConnectionBlock
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        # 初始化UnetSkipConnectionBlock
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # 初始化UnetSkipConnectionBlock
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # 初始化UnetSkipConnectionBlock
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        # 返回输入与Unet处理结果的和
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """构造一个具有跳过连接的Unet子模块

        参数:
            outer_nc (int) -- 外层卷积层的滤波器数量
            inner_nc (int) -- 内层卷积层的滤波器数量
            input_nc (int) -- 输入图像/特征的通道数
            submodule (UnetSkipConnectionBlock) -- 之前定义的子模块
            outermost (bool)    -- 如果这个模块是最外层模块
            innermost (bool)    -- 如果这个模块是最内层模块
            norm_layer          -- 归一化层
            user_dropout (bool) -- 如果使用dropout层
        """
        super(UnetSkipConnectionBlock, self).__init__()
        # 初始化
        self.outermost = outermost
        # 如果norm_layer是偏函数
        if type(norm_layer) == functools.partial:
            # 使用InstanceNorm2d
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            # 否则，使用BatchNorm2d
            use_bias = norm_layer == nn.InstanceNorm2d
        # 如果input_nc为None
        if input_nc is None:
            # 将outer_nc赋值给input_nc
            input_nc = outer_nc
        # 初始化卷积层
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        # 初始化激活函数
        downrelu = nn.LeakyReLU(0.2, True)
        # 初始化归一化层
        downnorm = norm_layer(inner_nc)
        # 初始化激活函数
        uprelu = nn.ReLU(True)
        # 初始化归一化层
        upnorm = norm_layer(outer_nc)

        # 如果是最外层
        if outermost:
            # 初始化卷积层
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # 初始化卷积层
            down = [downconv]
            # 初始化激活函数
            up = [uprelu, upconv, nn.Tanh()]
            # 将卷积层、激活函数和子模块连接起来
            model = down + [submodule] + up
        # 如果是最内层
        elif innermost:
            # 初始化卷积层
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # 初始化激活函数
            down = [downrelu, downconv]
            # 初始化激活函数
            up = [uprelu, upconv, upnorm]
            # 将卷积层、激活函数连接起来
            model = down + up
        else:
            # 初始化卷积层
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # 初始化激活函数
            down = [downrelu, downconv, downnorm]
            # 初始化激活函数
            up = [uprelu, upconv, upnorm]

            # 如果使用dropout
            if use_dropout:
                # 将卷积层、激活函数、子模块、激活函数和dropout连接起来
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                # 将卷积层、激活函数、子模块、激活函数连接起来
                model = down + [submodule] + up

        # 将卷积层连接起来
        self.model = nn.Sequential(*model)

    # 前向传播
    def forward(self, x):
        # 如果是最外层
        if self.outermost:
            # 返回处理结果
            return self.model(x)
        else:  # add skip connections
            # 将输入与处理结果连接起来
            return torch.cat([x, self.model(x)], 1)


# 定义N层判别器
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """构造一个PatchGAN判别器

        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层的滤波器数量
            n_layers (int)  -- 判别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        # 如果norm_layer是偏函数
        if type(norm_layer) == functools.partial:
            # 使用InstanceNorm2d
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 初始化卷积核大小和填充
        kw = 4
        padw = 1
        # 初始化序列
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        # 遍历n_layers次
        for n in range(1, n_layers):  # gradually increase the number of filters
            # 将nf_mult_prev赋值给nf_mult
            nf_mult_prev = nf_mult
            # 将nf_mult赋值为2的n次方和8的最小值
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 将nf_mult_prev赋值给nf_mult
        nf_mult_prev = nf_mult
        # 将nf_mult赋值为2的n_layers次方和8的最小值
        nf_mult = min(2 ** n_layers, 8)
        # 将卷积层、归一化层和激活函数连接起来
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        # 将卷积层连接起来
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    # 前向传播
    def forward(self, input):
        """Standard forward."""
        # 返回输入与处理结果的和
        return self.model(input)


# 定义1x1 PatchGAN判别器
class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构造一个1x1 PatchGAN判别器（像素GAN）

        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层的滤波器数量
            norm_layer      -- 归一化层
        """
        super(PixelDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters

        # 如果norm_layer是偏函数
        if type(norm_layer) == functools.partial:
            # 使用InstanceNorm2d
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 初始化序列
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        # 返回输入与处理结果的和
        return self.net(input)
