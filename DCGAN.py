from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML

# 为了可重复性设置随机种子
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 如果你想有一个不同的结果使用这行代码
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 数据集根目录
dataroot = "face/"
#fake_samples_outputroot
fake_outroot = "testimg"
#G_models_output_root
G_modelsroot = "G_models"
# 数据加载器能够使用的进程数量
workers = 2

# 训练时的批大小
batch_size = 32

# 训练图片的大小，所有的图片给都将改变到该大小
# 转换器使用的大小.
image_size = 64

# 训练图片的通道数，彩色图片是3
nc = 3

# 本征向量z的大小(生成器的输入大小)
nz = 100

# 生成器中特征图大小
ngf = 64

# 判别器中特征图大小
ndf = 64

# 训练次数
num_epochs = 10

# 优化器学习率
lr = 0.0002

# Adam优化器的Beta1超参
beta1 = 0.5

# 可利用的GPU数量，使用0将运行在CPU模式。
ngpu = 1

# 我们能够使用我们创建的数据集图片文件夹了
# 创建数据集
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 决定我们在哪个设备上运行
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 展示一些训练图片
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# 在netG和netD上调用的自定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 生成器代码

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入是 Z, 对Z进行卷积
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入特征图大小. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入特征图大小. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入特征图大小. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输入特征图大小. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输入特征图大小. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
# 创建生成器
netG = Generator(ngpu).to(device)

# 如果期望使用多个GPU，设置一下。
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 使用权重初始化函数 weights_init 去随机初始化所有权重
#  mean=0, stdev=0.2.
netG.apply(weights_init)

# 输出该模型
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入大小 (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入大小. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# 创建判别器
netD = Discriminator(ngpu).to(device)

# 如果期望使用多GPU，设置一下
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 使用权重初始化函数 weights_init 去随机初始化所有权重
#  mean=0, stdev=0.2.
netD.apply(weights_init)

# 输出该模型
print(netD)

# 初始化 BCE损失函数
criterion = nn.BCELoss()

# 创建一个批次的本征向量用于可视化生成器训练的过程。
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 建立一个在训练中使用的真实和假的标记
real_label = 1
fake_label = 0

# 为G和D都设置Adam优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# 训练循环

# 保存跟踪进度的列表
img_list = []
G_losses = []
D_losses = []
iters = 0
img_num = 0
print("Starting Training Loop...")
# 每个epoh
for epoch in range(num_epochs):
    # 数据加载器中的每个批次
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) 更新 D 网络: 最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 使用所有真实样本批次训练
        netD.zero_grad()
        # 格式化批
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # 通过D向前传递真实批次
        output = netD(real_cpu).view(-1)
        # 对所有真实样本批次计算损失
        errD_real = criterion(output, label)
        # 计算后向传递中D的梯度
        errD_real.backward()
        D_x = output.mean().item()

        ## 使用所有假样本批次训练
        # 生成本征向量批次
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 使用生成器G生成假图片
        fake = netG(noise)
        label.fill_(fake_label)
        # 使用判别器分类所有的假批次样本
        output = netD(fake.detach()).view(-1)
        # 计算判别器D的损失对所有的假样本批次
        errD_fake = criterion(output, label)
        # 对这个批次计算梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 把所有真样本和假样本批次的梯度加起来
        errD = errD_real + errD_fake
        # 更新判别器D
        optimizerD.step()

        ############################
        # (2) 更新 G 网络: 最大化 log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 假样本的标签对于生成器成本是真的
        # 因为我们之更新了D，通过D执行所有假样本批次的正向传递
        output = netD(fake).view(-1)
        # 基于这个输出计算G的损失
        errG = criterion(output, label)
        # 为生成器计算梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新生成器G
        optimizerG.step()

        # 输出训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 为以后画损失图，保存损失
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 检查生成器generator做了什么，通过保存的fixed_noise通过G的输出
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()

            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            vutils.save_image(img_list[img_num],'%s/fake_sample_%d.png'%(fake_outroot,img_num))
            img_num += 1

        iters += 1
        torch.save(netG.state_dict(), "%s/netG_%d.pth" %(G_modelsroot,epoch))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

