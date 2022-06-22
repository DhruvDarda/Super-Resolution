import streamlit as st
from PIL import Image
import io
import os
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt


Tensor = torch.Tensor


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


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(
                filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(
            channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(
            filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


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


def load_image(uploaded_file):
    image_data = uploaded_file.getvalue()
    # st.image(image_data)
    return Image.open(io.BytesIO(image_data)).convert('RGB')


def to_np(x):
    return x.data.cpu().numpy()


def plot_result(real_image, gen_image, fig_size=(8, 4)):
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    imgs = [to_np(real_image), to_np(gen_image)]
    labels = ['Low-Resolution Image', 'Super-Resolution Image']
    for i, (ax, img) in enumerate(zip(axes.flatten(), imgs)):
        ax.axis('off')
        img = img.squeeze()
        img = (((img - img.min()) * 255) / (img.max() - img.min())
               ).transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
        ax.set_title(labels[i])
    plt.subplots_adjust(wspace=0, hspace=0)

    title = "SR vs HR"
    fig.text(0.5, 0.04, title, ha='center')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im


def main():
    #PATH = os.getcwd()
    st.title('Image Super-Resolution demo')
    hr_height = 256
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    lr_transform = transforms.Compose(
        [
            transforms.Resize(
                (hr_height // 4, hr_height // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    uploaded_file = st.file_uploader(label='Pick an image to improve')
    if uploaded_file is not None:
        real_image = load_image(uploaded_file)
        convert = transforms.Resize((hr_height, hr_height), Image.BICUBIC)

        with st.container():
            st.markdown("<h4 style='text-align: center; color: red;'>Uploaded Image:</h4>",
                        unsafe_allow_html=True)
            with open("style.css") as f:
                st.markdown('<style>{}</style>'.format(f.read()),
                            unsafe_allow_html=True)
                st.image(convert(real_image))
            real_image = lr_transform(real_image)
            real_image = torch.unsqueeze(real_image, dim=0)
            real_image = Variable(real_image.type(Tensor))

            generator_select = st.selectbox(
                'Model', ("SRGAN", "ESRGAN", "Compare"))
            if generator_select == 'SRGAN':
                generator = GeneratorResNet()
                generator.load_state_dict(torch.load(
                    'generator_srgan.pt', map_location=torch.device('cpu')))

                if st.button('Generate'):
                    gen_image = generator(real_image)
                    result_img = plot_result(real_image, gen_image)
                    st.image(result_img)

            elif generator_select == 'ESRGAN':
                generator = GeneratorResNet()
                generator.load_state_dict(torch.load(
                    'generator_srgan.pth', map_location=torch.device('cpu')))

                if st.button('Generate'):
                    gen_image = generator(real_image)
                    result_img = plot_result(real_image, gen_image)
                    st.image(result_img)

            else:
                if st.button('Generate'):
                    st.write('ESRGAN Outut')
                    generator = GeneratorResNet()
                    generator.load_state_dict(torch.load(
                        'generator_srgan.pth', map_location=torch.device('cpu')))
                    gen_image = generator(real_image)
                    result_img = plot_result(real_image, gen_image)
                    st.image(result_img)

                    st.write('SRGAN Outut')
                    generator = GeneratorResNet()
                    generator.load_state_dict(torch.load(
                        'generator_srgan.pt', map_location=torch.device('cpu')))
                    gen_image = generator(real_image)
                    result_img = plot_result(real_image, gen_image)
                    st.image(result_img)


if __name__ == '__main__':
    main()
