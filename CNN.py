import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from common_tools import transform_invert   # set_seed 有问题
from random import seed

# set_seed(3)  # 设置随机种子
seed(3)

# ================================= load img ==================================
# path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "lena.png")
path_img = f"D:\屏保.jpg"
    #os.path.abspath(__file__)获取这个py文件绝对路径
    #os.path.dirname(os.path.abspath(__file__)获取这个py文件的目录  如E：\\abs\\ni.py的目录就是E:\\abs
print(path_img)
img = Image.open(path_img).convert('RGB')  # 0~255
img.show()
img = img.rotate(-90,expand = True)   #顺时针旋转
img.show()   #展示出来的图像大小尺寸是跟之前的一样，缺少了信息

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
# 添加 batch 维度
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================

# ================ 2d
flag = 1
# flag = 0
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
    # 初始化卷积层权值
    nn.init.xavier_normal_(conv_layer.weight.data)
 # nn.init.xavier_uniform_(conv_layer.weight.data)
    # calculation
    img_conv = conv_layer(img_tensor)

# ================ transposed
# flag = 1
flag = 0
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)   # input:(input_channel, output_channel, size)
    # 初始化网络层的权值
    nn.init.xavier_normal_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================================= visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()