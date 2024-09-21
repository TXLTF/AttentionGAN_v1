import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """这个类是一个抽象基类（ABC），用于模型。
    要创建一个子类，你需要实现以下五个函数：
        -- <__init__>:                     初始化类；首先调用BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                      产生中间结果。
        -- <optimize_parameters>:          计算损失、梯度并更新网络权重。
        -- <modify_commandline_options>:   (optionally) 添加模型特定的选项并设置默认选项。
    """

    def __init__(self, opt):
        """初始化BaseModel类。

        参数:
            opt (Option class)-- 存储所有实验标志；需要是BaseOptions的子类

        在创建自定义类时，你需要实现自己的初始化。
        在这个函数中，你应该首先调用<BaseModel.__init__(self, opt)>
        然后，你需要定义四个列表：
            -- self.loss_names (str list):         指定要绘制和保存的训练损失。
            -- self.model_names (str list):        指定要显示和保存的图像。
            -- self.visual_names (str list):       定义我们在训练中使用的网络。
            -- self.optimizers (optimizer list):   定义并初始化优化器。你可以为每个网络定义一个优化器。如果两个网络同时更新，你可以使用itertools.chain来分组它们。查看cycle_gan_model.py的示例。
        """
        # 初始化模型   
        self.opt = opt
        # 获取GPU ID
        self.gpu_ids = opt.gpu_ids
        # 是否是训练模型
        self.isTrain = opt.isTrain
        # 获取设备名称：CPU或GPU
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        # 保存检查点目录
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # 如果预处理不是scale_width，则启用cudnn.benchmark
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        # 初始化损失名称列表
        self.loss_names = []
        # 初始化模型名称列表
        self.model_names = []
        # 初始化可视化名称列表
        self.visual_names = []
        # 初始化优化器列表
        self.optimizers = []
        # 初始化图像路径列表
        self.image_paths = []
        # 用于学习率策略'plateau'的度量
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    # 修改命令行选项
    def modify_commandline_options(parser, is_train):
        """添加新的模型特定选项，并重写现有选项的默认值。

        参数:
            parser          -- 原始选项解析器
            is_train (bool) -- 是否是训练阶段或测试阶段。你可以使用这个标志来添加训练特定或测试特定的选项。

        返回:
            修改后的解析器。
        """
        return parser

    @abstractmethod
    # 设置输入
    def set_input(self, input):
        """从数据加载器中解包输入数据，并执行必要的预处理步骤。

        参数:
            input (dict): 包括数据本身及其元数据信息。
        """
        pass

    # 前向传播
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    # 优化参数
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用"""
        pass

    # 加载网络并打印
    def setup(self, opt):
        """加载网络并打印；创建调度器

        参数:
            opt (Option class) -- 存储所有实验标志；需要是BaseOptions的子类
        """
        # 如果训练，则创建调度器
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # 如果不需要训练，或者继续训练
        if not self.isTrain or opt.continue_train:
            # 加载网络
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        # 打印网络
        self.print_networks(opt.verbose)

    # 评估
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            # 如果name是字符串
            if isinstance(name, str):
                # 获取网络
                net = getattr(self, 'net' + name)
                # 设置为评估模式
                net.eval()

    # 测试
    def test(self):
        """Forward function used in test time.

        参数:
            This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
            这个函数在no_grad()中包装了<forward>函数，所以我们不会保存中间步骤用于反向传播
            它还调用<compute_visuals>来生成额外的可视化结果
        """
        # 使用torch.no_grad()，这样我们不会保存中间步骤用于反向传播
        with torch.no_grad():
            # 前向传播
            self.forward()
            # 计算可视化
            self.compute_visuals()

    # 计算可视化
    def compute_visuals(self):
        """计算额外的输出图像，用于visdom和HTML可视化"""
        pass

    # 获取图像路径
    def get_image_paths(self):
        """返回用于加载当前数据的图像路径"""
        return self.image_paths

    # 更新学习率
    def update_learning_rate(self):
        """更新所有网络的学习率；在每个epoch结束时调用"""
        for scheduler in self.schedulers:
            # 如果学习率策略是plateau
            if self.opt.lr_policy == 'plateau':
                # 更新学习率
                scheduler.step(self.metric)
            else:
                # 更新学习率
                scheduler.step()

        # 获取学习率
        lr = self.optimizers[0].param_groups[0]['lr']
        # 打印学习率
        print('learning rate = %.7f' % lr)

    # 获取当前可视化    
    def get_current_visuals(self):
        """返回可视化图像。train.py将使用visdom显示这些图像，并将图像保存到HTML"""
        # 初始化可视化字典
        visual_ret = OrderedDict()
        # 遍历可视化名称列表
        for name in self.visual_names:
            # 如果name是字符串
            if isinstance(name, str):
                # 获取属性
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # 获取当前损失
    def get_current_losses(self):
        """返回训练损失/错误。train.py将在控制台上打印这些错误，并将它们保存到一个文件中"""
        # 初始化损失字典
        errors_ret = OrderedDict()
        # 遍历损失名称列表
        for name in self.loss_names:
            # 如果name是字符串
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    # 保存网络
    def save_networks(self, epoch):
        """将所有网络保存到磁盘。

        参数:
            epoch (int) -- 当前epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        # 遍历模型名称列表
        for name in self.model_names:
            # 如果name是字符串  
            if isinstance(name, str):
                # 保存网络
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                # 如果GPU ID大于0且GPU可用
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # 保存网络
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # 将网络移动到GPU
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # 修复InstanceNorm检查点的不兼容性（0.4之前）
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复InstanceNorm检查点的不兼容性（0.4之前）"""
        # 获取键
        key = keys[i]
        # 如果i+1等于键的长度
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            # 如果模块是InstanceNorm且键是running_mean或running_var
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                # 如果属性为None
                if getattr(module, key) is None:
                    # 删除键
                    state_dict.pop('.'.join(keys))
            # 如果模块是InstanceNorm且键是num_batches_tracked
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                # 删除键
                state_dict.pop('.'.join(keys))
        else:
            # 递归调用
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # 加载网络
    def load_networks(self, epoch):
        """Load all the networks from the disk.

        参数:
            epoch (int) -- 当前epoch；用于文件名 '%s_net_%s.pth' % (epoch, name)
        """
        # 遍历模型名称列表
        for name in self.model_names:
            # 如果name是字符串
            if isinstance(name, str):
                # 获取加载文件名
                load_filename = '%s_net_%s.pth' % (epoch, name)
                # 获取加载路径
                load_path = os.path.join(self.save_dir, load_filename)
                # 获取网络
                net = getattr(self, 'net' + name)
                # 如果net是DataParallel
                if isinstance(net, torch.nn.DataParallel):
                    # 获取net的module
                    net = net.module
                # 打印加载路径
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                # 加载网络
                state_dict = torch.load(load_path, map_location=str(self.device))
                # 删除_metadata
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # 修复InstanceNorm检查点的不兼容性（0.4之前）
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    # 递归调用
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # 加载网络
                net.load_state_dict(state_dict)

    # 打印网络
    def print_networks(self, verbose):
        """打印网络中参数的总数（如果verbose）网络架构

        参数:
            verbose (bool) -- 如果verbose: 打印网络架构
        """
        print('---------- Networks initialized -------------')
        # 遍历模型名称列表  
        for name in self.model_names:
            # 如果name是字符串
            if isinstance(name, str):
                # 获取网络
                net = getattr(self, 'net' + name)
                # 初始化参数数量
                num_params = 0
                # 遍历网络的参数
                for param in net.parameters():
                    # 累加参数数量
                    num_params += param.numel()
                # 如果verbose为真
                if verbose:
                    # 打印网络
                    print(net)
                # 打印参数数量
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # 设置是否需要梯度
    def set_requires_grad(self, nets, requires_grad=False): 
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        参数:
            nets (network list)   -- 一个网络列表
            requires_grad (bool)  -- 网络是否需要梯度
        """
        """设置所有网络的requires_grad=False,以避免不必要的计算
        参数:
            nets (网络列表) -- 一个网络列表
            requires_grad (布尔值) -- 网络是否需要梯度
        """
        # 如果nets不是列表，则将其转换为列表
        if not isinstance(nets, list):
            nets = [nets]
        # 遍历网络列表
        for net in nets:
            # 如果网络不为空
            if net is not None:
                # 遍历网络的参数
                for param in net.parameters():
                    param.requires_grad = requires_grad
