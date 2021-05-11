import torch
import torch.nn as nn


def conv1x1(in_planes, planes):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=1, bias=False),
    )


def con15x1(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=planes//4, kernel_size=15, stride=stride, padding=7),
        nn.BatchNorm1d(planes//4)
    )


def con17x1(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=planes//4, kernel_size=1,  stride=1, bias=False),
        nn.BatchNorm1d(planes // 4),
        nn.LeakyReLU(0.2),
        nn.Conv1d(in_channels=planes // 4, out_channels=planes // 4, kernel_size=17, stride=stride,
                  padding=8),
        nn.BatchNorm1d(planes // 4)
    )


def con19x1(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=planes//4, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(planes // 4),
        nn.LeakyReLU(0.2),
        nn.Conv1d(in_channels=planes // 4, out_channels=planes // 4, kernel_size=19, stride=stride,
                  padding=9),
        nn.BatchNorm1d(planes // 4)
    )


def con21x1(in_planes, planes, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=planes//4, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(planes // 4),
        nn.LeakyReLU(0.2),
        nn.Conv1d(in_channels=planes // 4, out_channels=planes // 4, kernel_size=21, stride=stride,
                  padding=10),
        nn.BatchNorm1d(planes // 4)
    )


class IncResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(IncResBlock, self).__init__()
        self.conv1x1 = conv1x1(in_planes=in_planes, planes=planes)
        self.conv15x1 = con15x1(in_planes=in_planes, planes=planes, stride=stride)
        self.conv17x1 = con17x1(in_planes=in_planes, planes=planes, stride=stride)
        self.conv19x1 = con19x1(in_planes=in_planes, planes=planes, stride=stride)
        self.conv21x1 = con21x1(in_planes=in_planes, planes=planes, stride=stride)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(planes//4)

    def forward(self, signal):
        residual = (self.conv1x1(signal))
        c1 = self.conv15x1(signal)
        c2 = self.conv17x1(signal)
        c3 = self.conv19x1(signal)
        c4 = self.conv21x1(signal)
        out = torch.cat([c1, c2, c3, c4], dim=1)
        out += residual
        return residual


class IncUNet(nn.Module):
    def __init__(self, in_shape):
        super(IncUNet, self).__init__()
        in_channels = in_shape
        self.e1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, ),
            IncResBlock(64, 64))
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            IncResBlock(128, 128))
        self.e2add = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128))
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256, 256))
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))
        self.e4add = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512))
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.e6add = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512))

        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512))

        self.d1 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.d2 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.d3 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            IncResBlock(512, 512))

        self.d4 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.d5 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.d6 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            IncResBlock(512, 512))

        self.d7 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(1024, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm1d(256),
            IncResBlock(256, 256))

        self.d8 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128))

        self.d9 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128))

        self.d10 = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64))

        self.out_l = nn.Sequential(
            nn.LeakyReLU(0.2, ),
            nn.ConvTranspose1d(128, in_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, x):
        en1 = self.e1(x)

        en2 = self.e2(en1)

        en2add = self.e2add(en2)

        en3 = self.e3(en2add)

        en4 = self.e4(en3)

        en4add = self.e4add(en4)

        en5 = self.e5(en4add)

        en6 = self.e6(en5)

        en6add = self.e6add(en6)

        en7 = self.e7(en6add)
        # print("ee7的维度：", en7.shape)
        en8 = self.e8(en7)

        de1_ = self.d1(en8)
        # print("de1_的维度：", de1_.shape)
        de1 = torch.cat([en7, de1_], 1)
        # print("de1的维度：", de1.shape)
        de2_ = self.d2(de1)
        # print("de2_的维度：", de2_.shape)
        de2 = torch.cat([en6add, de2_], 1)
        # print("de2的维度：", de2.shape)
        de3_ = self.d3(de2)
        # print("de3_的维度：", de3_.shape)
        de3 = torch.cat([en6, de3_], 1)
        # print("de3的维度：", de3.shape)
        de4_ = self.d4(de3)
        # print("de4_的维度：", de4_.shape)
        de4 = torch.cat([en5, de4_], 1)
        # print("de4的维度：", de4.shape)
        de5_ = self.d5(de4)
        # print("de5_的维度：", de5_.shape)
        de5 = torch.cat([en4add, de5_], 1)
        # print("de5的维度：", de5.shape)
        de6_ = self.d6(de5)

        de6 = torch.cat([en4, de6_], 1)

        de7_ = self.d7(de6)
        # print("de7_的维度：", de7_.shape)
        de7 = torch.cat([en3, de7_], 1)
        # print("de7的维度：", de7.shape)
        de8_ = self.d8(de7)
        # print("de8_的维度：", de8_.shape)
        de8 = torch.cat([en2add, de8_], 1)
        # print("de8的维度：", de8.shape)
        de9_ = self.d9(de8)

        de9 = torch.cat([en2, de9_], 1)

        de10_ = self.d10(de9)

        de10 = torch.cat([en1, de10_], 1)

        out = self.out_l(de10)
        return out

