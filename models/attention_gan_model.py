import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AttentionGANModel(BaseModel):
    @staticmethod
    # 修改命令行选项
    def modify_commandline_options(parser, is_train=True):
        """修改命令行选项

        参数:
            parser - - 原始选项解析器
            is_train - - 是否为训练阶段
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout
        # 如果训练，则添加循环损失和身份映射损失的权重
        if is_train:
            # 添加循环损失（A -> B -> A）的权重
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss (A -> B -> A)')
            # 添加循环损失（B -> A -> B）的权重
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            # 添加身份映射损失的权重
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # 返回修改后的解析器
        return parser

    def __init__(self, opt):
        """初始化模型

        参数:
            opt (Option class) -- 存储所有实验标志；需要是BaseOptions的子类
        """
        # 初始化模型
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A',
                           'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'o3_b', 'o4_b', 'o5_b', 'o6_b', 'o7_b', 'o8_b', 'o9_b', 'o10_b',
                          'a1_b', 'a2_b', 'a3_b', 'a4_b', 'a5_b', 'a6_b', 'a7_b', 'a8_b', 'a9_b', 'a10_b', 'i1_b', 'i2_b', 'i3_b', 'i4_b', 'i5_b',
                          'i6_b', 'i7_b', 'i8_b', 'i9_b']
        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'o3_a', 'o4_a', 'o5_a', 'o6_a', 'o7_a', 'o8_a', 'o9_a', 'o10_a',
                          'a1_a', 'a2_a', 'a3_a', 'a4_a', 'a5_a', 'a6_a', 'a7_a', 'a8_a', 'a9_a', 'a10_a', 'i1_a', 'i2_a', 'i3_a', 'i4_a', 'i5_a',
                          'i6_a', 'i7_a', 'i8_a', 'i9_a']
        # 如果使用身份损失，则还可视化idt_B=G_A(B)和idt_A=G_A(B)
        # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # 如果保存磁盘，则只保存一些图像
        if self.opt.saveDisk:
            # 将A和B的可视化名称组合在一起
            self.visual_names = ['real_A', 'fake_B',
                                 'a10_b', 'real_B', 'fake_A', 'a10_a']
        else:
            # 将A和B的可视化名称组合在一起
            # combine visualizations for A and B
            self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        # 如果训练，则保存生成器和判别器
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            # 如果未训练，则只保存生成器
            self.model_names = ['G_A', 'G_B']

        # 定义网络（生成器和判别器）
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # 定义生成器G_A
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 定义生成器G_B
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # 定义判别器D_A
        if self.isTrain:  # define discriminators
            # 定义判别器D_A
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # 定义判别器D_B
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # 如果在训练模式，还会初始化图像缓冲池、损失函数和优化器
        if self.isTrain:
            # 如果输入和输出图像具有相同的通道数，则使用身份映射
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            # 创建图像缓冲池，用于存储之前生成的图像
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # 定义GAN损失函数
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # 定义循环损失函数
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 初始化优化器
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(
            ), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(
            ), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # 将优化器添加到优化器列表中
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # 用于处理数据加载器提供的输入数据
    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input (dict): 包含数据本身及其元数据信息。

        The option 'direction' can be used to swap domain A and domain B.
        """
        # 根据方向选择输入的图像
        AtoB = self.opt.direction == 'AtoB'
        # 将输入的图像转换为张量并移动到指定的设备
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # 将输入的图像路径添加到图像路径列表中
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # 执行模型的前向传播，生成假图像和重建图像。
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # 生成假图像和重建图像
        self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
            self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
            self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(
                self.real_A)  # G_A(A)
        self.rec_A, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = self.netG_B(
                self.fake_B)   # G_B(G_A(A))
        self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
            self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
            self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(
                self.real_B)  # G_B(B)
        self.rec_B, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, _, _, _ = self.netG_A(
                self.fake_A)   # G_A(G_B(B))

    # 计算判别器的损失
    def backward_D_basic(self, netD, real, fake):
        """计算判别器的GAN损失

        参数:
            netD (network) -- 判别器D
            real (tensor array) -- 真实图像
            fake (tensor array) -- 生成器生成的图像
        返回:
            判别器的损失
        我们还将调用loss_D.backward()来计算梯度。
        """

        # Real
        # 计算真实图像的判别器损失
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        # 计算假图像的判别器损失
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        # 计算判别器的总损失
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    # 计算判别器D_A的损失
    def backward_D_A(self):
        """计算判别器D_A的GAN损失"""
        # 从图像缓冲池中查询假图像
        fake_B = self.fake_B_pool.query(self.fake_B)
        # 计算判别器的损失
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    # 计算判别器D_B的损失
    def backward_D_B(self):
        """计算判别器D_B的GAN损失"""
        # 从图像缓冲池中查询假图像
        fake_A = self.fake_A_pool.query(self.fake_A)
        # 计算判别器的损失
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # 计算生成器的损失
    def backward_G(self):
        """计算生成器G_A和G_B的损失"""
        # 获取身份损失的权重
        lambda_idt = self.opt.lambda_identity
        # 获取循环损失的权重
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        # 如果使用身份损失，则计算身份损失
        if lambda_idt > 0:
            # G_A应该在输入真实B时是恒等映射：||G_A(B) - B||
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # 计算G_A(B)
            self.idt_A, _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _ = self.netG_A(self.real_B)
            # 计算G_A(B)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B应该在输入真实A时是恒等映射：||G_B(A) - A||
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # 计算G_B(A)
            self.idt_B, _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _, _, \
                _, _, _, _, _, _, _, _, _ = self.netG_B(self.real_A)
            # 计算G_B(A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            # 如果未使用身份损失，则将损失设置为0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        # 计算生成器G_A的损失
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        # 计算生成器G_B的损失
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # 计算前向循环损失
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        # 计算后向循环损失
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        # 计算生成器的总损失
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + \
            self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # 计算生成器的梯度
        self.loss_G.backward()

    # 优化网络参数
    def optimize_parameters(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用"""
        # forward
        # 前向传播，生成假图像和重建图像
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # 设置判别器不需要梯度
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        # 将生成器G_A和G_B的梯度设置为0
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # 计算生成器G_A和G_B的梯度
        self.backward_G()             # calculate gradients for G_A and G_B
        # 更新生成器G_A和G_B的权重
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        # 设置判别器需要梯度
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        # 将判别器D_A和D_B的梯度设置为0
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        # 计算判别器D_A和D_B的梯度
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        # 更新判别器D_A和D_B的权重
        self.optimizer_D.step()  # update D_A and D_B's weights''
