import os
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


def to_np(x):
    return x.data.cpu().numpy()


current = os.getcwd()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = GeneratorResNet()
generator.load_state_dict(torch.load(
    current+'\generator_256.pt', map_location=torch.device(device)))

hr_height = 256
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def add_margin(pil_img):
    width, height = pil_img.size
    left = (60-width)//2
    top = (60-height)//2
    result = Image.new(pil_img.mode, (60, 60), (0, 0, 0))
    result.paste(pil_img, (left, top))
    return result


def retain_shape(pil_img):
    width, height = pil_img.size
    shape = max(width, height)
    left = (shape - width)//2
    top = (shape - height)//2
    result = Image.new(pil_img.mode, (shape, shape), (0, 0, 0))
    result.paste(pil_img, (left, top))
    return result


lr_transform = transforms.Compose(
    [
        transforms.Resize(
            (hr_height // 4, hr_height // 4), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

hr_transform = transforms.Compose(
    [
        transforms.Resize((hr_height, hr_height), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

gen_transform = transforms.ToPILImage()

source_dir = os.path.join(current, 'LR_Images')


for img in os.listdir('LR_Images/'):
    # img = 'test3.jpg' #files.upload()
    real_image = Image.open('LR_Images/' + img).convert('RGB')
    o_height, o_width = real_image.size
    if o_height >= 25600 and o_width >= 25600:
        gen_image_pil = real_image
        print('Already an HR Image')
    else:
        if o_height < 64 or o_width < 64:
            real_image = add_margin(real_image)
        # if o_height != o_width:
            #real_image = retain_shape(real_image)
        recon_image = hr_transform(real_image)
        real_image = lr_transform(real_image)
        real_image = torch.unsqueeze(real_image, dim=0)
        real_image = Variable(real_image.type(torch.Tensor))

        gen_image = generator(real_image)
        gen_imagenp = to_np(gen_image.squeeze())
        gen_imagenp = (((gen_imagenp - gen_imagenp.min()) * 255) / (
            gen_imagenp.max() - gen_imagenp.min())).transpose(1, 2, 0).astype(np.uint8)
        gen_image_pil = Image.fromarray(gen_imagenp)
        print('Super-Resolved the image successfully!')

    name = 'SR_Images/' + img
    gen_image_pil.save(name)
