#!/usr/bin/python3

import argparse
import itertools
import sys
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import torchvision

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import print_network
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200,
                    help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1,
                    help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/',
                    help='root directory of the dataset')
parser.add_argument('--save_name', type=str, default='ar_neutral2happiness')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256,
                    help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3,
                    help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3,
                    help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--lambda_cycle', type=int, default=10)
parser.add_argument('--lambda_identity', type=int, default=0)
parser.add_argument('--lambda_pixel', type=int, default=1)
parser.add_argument('--lambda_reg', type=float, default=1e-6)

parser.add_argument('--gan_curriculum', type=int, default=10,
                    help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01,
                    help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5,
                    help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

# 解析命令行参数，并将结果存储在opt对象中
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
# 创建生成器和判别器网络实例
netG_A2B = Generator()
netG_B2A = Generator()
netD_A = Discriminator()
netD_B = Discriminator()

print('---------- Networks initialized -------------')
print_network(netG_A2B)
print_network(netG_B2A)
print_network(netD_A)
print_network(netD_B)
print('-----------------------------------------------')

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# 使用自定义的权重初始化方法初始化生成器和判别器的权重
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
# 定义GAN损失函数
criterion_GAN = torch.nn.MSELoss()
# 定义循环损失函数
criterion_cycle = torch.nn.L1Loss()
# 定义身份损失函数
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
# 创建优化器和学习率调度器
# 创建生成器优化器
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
# 创建判别器优化器
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))

# 创建学习率调度器
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
# 创建输入张量
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# 创建真实标签张量
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# 创建虚假标签张量
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# 创建缓冲区
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
# 创建数据加载器
transforms_ = [transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
# 创建日志记录器
logger = Logger(opt.n_epochs, len(dataloader))

###### Training ######
# 训练循环
for epoch in range(opt.epoch, opt.n_epochs):
    # 遍历数据加载器
    for i, batch in enumerate(dataloader):
        # 设置模型输入
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        # 生成器优化器梯度清零
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # 生成器A2B(B)应该等于B，如果输入是真实的B
        same_B, _, _ = netG_A2B(real_B)
        # 计算身份损失
        loss_identity_B = criterion_identity(
            same_B, real_B)*opt.lambda_identity
        # G_B2A(A) should equal A if real A is fed
        # 生成器B2A(A)应该等于A，如果输入是真实的A
        same_A, _, _ = netG_B2A(real_A)
        # 计算身份损失
        loss_identity_A = criterion_identity(
            same_A, real_A)*opt.lambda_identity

        # GAN loss
        # 生成器A2B(A)应该等于B，如果输入是真实的A
        fake_B, mask_B, temp_B = netG_A2B(real_A)
        # 生成器B2A(B)应该等于A，如果输入是真实的B
        recovered_A, _, _ = netG_B2A(fake_B)
        # 预测虚假的B
        pred_fake_B = netD_B(fake_B)

        # 计算循环损失
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
        # 计算GAN损失
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
        # 计算像素损失
        loss_pix_A = criterion_identity(fake_B, real_A)

        # 生成器B2A(B)应该等于A，如果输入是真实的B
        fake_A, mask_A, temp_A = netG_B2A(real_B)
        # 生成器A2B(A)应该等于B，如果输入是真实的A
        recovered_B, _, _ = netG_A2B(fake_A)
        # 预测虚假的A
        pred_fake_A = netD_A(fake_A)

        # 计算循环损失
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)
        # 计算GAN损失
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)
        # 计算像素损失
        loss_pix_B = criterion_identity(fake_A, real_B)

        # 计算正则化损失    
        loss_reg_A = opt.lambda_reg * (
            torch.sum(torch.abs(mask_A[:, :, :, :-1] - mask_A[:, :, :, 1:])) +
            torch.sum(torch.abs(mask_A[:, :, :-1, :] - mask_A[:, :, 1:, :])))

        loss_reg_B = opt.lambda_reg * (
            torch.sum(torch.abs(mask_B[:, :, :, :-1] - mask_B[:, :, :, 1:])) +
            torch.sum(torch.abs(mask_B[:, :, :-1, :] - mask_B[:, :, 1:, :])))

        # 计算总损失
        if epoch < opt.gan_curriculum:
            # 使用课程学习GAN
            rate = opt.starting_rate
            print('using curriculum gan')
        else:
            # 使用正常GAN
            rate = opt.default_rate
            print('using normal gan')

        # 计算生成器损失
        loss_G = ((loss_GAN_A2B + loss_GAN_B2A)*0.5 + (loss_reg_A + loss_reg_B)) * (1.-rate) + \
            ((loss_cycle_ABA + loss_cycle_BAB)*opt.lambda_cycle +
             (loss_pix_B+loss_pix_A)*opt.lambda_pixel) * rate

        # 反向传播生成器损失
        loss_G.backward()
        # 更新生成器参数
        optimizer_G.step()
        ###################################
        # 判别器优化器梯度清零
        optimizer_D.zero_grad()

        # 计算判别器损失
        # Real loss
        pred_real_A = netD_A.forward(real_A)
        # 计算真实损失
        loss_D_real_A = criterion_GAN(pred_real_A, target_real)

        # Fake loss
        # 将虚假的A推入缓冲区
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        # 计算虚假损失
        pred_fake_A = netD_A.forward(fake_A.detach())
        # 计算真实损失
        loss_D_real_A = criterion_GAN(pred_real_A, target_real)

        # Real loss
        # 计算真实损失
        pred_real_B = netD_B.forward(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, target_real)

        # Fake loss
        # 将虚假的B推入缓冲区
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        # 计算虚假损失
        pred_fake_B = netD_B.forward(fake_B.detach())
        # 计算虚假损失
        loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)

        # Total loss
        # 计算判别器损失
        loss_D = (loss_D_real_B + loss_D_fake_B +
                  loss_D_real_A + loss_D_fake_A)*0.25
        # 反向传播判别器损失
        loss_D.backward()
        # 更新判别器参数
        optimizer_D.step()

        # 打印损失
        print('Epoch [%d/%d], Batch [%d/%d], loss_D: %.4f, loss_G: %.4f' %
              (epoch+1, opt.n_epochs, i+1, len(dataloader), loss_D.data[0], loss_G.data[0]))
        print('loss_GAN_A2B: %.4f, loss_GAN_B2A: %.4f, loss_cycle_ABA: %.4f, loss_cycle_BAB: %.4f, loss_identity_A: %.4f, loss_identity_B: %.4f, loss_pix_A: %.4f, loss_pix_B: %.4f' % (loss_GAN_A2B.data[0],
                                                                                                                                                                                        loss_GAN_B2A.data[0], loss_cycle_ABA.data[0], loss_cycle_BAB.data[0], loss_identity_A.data[0], loss_identity_B.data[0], loss_pix_A.data[0], loss_pix_B.data[0]))

        save_path = '%s/%s' % (opt.save_name, 'training')
        # 如果保存路径不存在，则创建
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_image(torch.cat([
            real_A.data.cpu()[0] * 0.5 + 0.5,
            mask_B.data.cpu()[0],
            fake_B.data.cpu()[0]*0.5+0.5, temp_B.data.cpu()[0]*0.5+0.5], 2),
            '%s/%04d_%04d_progress_B.png' % (save_path, epoch+1, i+1))

        save_image(torch.cat([
            real_B.data.cpu()[0] * 0.5 + 0.5,
            mask_A.data.cpu()[0],
            fake_A.data.cpu()[0]*0.5+0.5, temp_A.data.cpu()[0]*0.5+0.5], 2),
            '%s/%04d_%04d_progress_A.png' % (save_path, epoch+1, i+1))

    # 更新学习率
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # 保存生成器和判别器模型
    torch.save(netG_A2B.state_dict(), '%s/%s' %
               (opt.save_name, 'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), '%s/%s' %
               (opt.save_name, 'netG_B2A.pth'))
    # 保存判别器模型
    torch.save(netD_A.state_dict(), '%s/%s' % (opt.save_name, 'netD_A.pth'))
    torch.save(netD_B.state_dict(), '%s/%s' % (opt.save_name, 'netD_B.pth'))
