
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, init_num_classes=100):
        super().__init__()

        self.in_channels = 64
        self.init_num_classes = init_num_classes
        self.block_expansion = block.expansion


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block_expansion, init_num_classes)

    def _add_class(self, add_num_classed):
        self.fc_new = nn.Linear(512 * self.block_expansion, self.init_num_classes+add_num_classed)
        nn.init.zeros_(self.fc_new.weight)
        nn.init.zeros_(self.fc_new.bias)
        self.fc_new.weight[:self.init_num_classes,:] = self.fc.weight
        self.fc_new.bias[:self.init_num_classes,:] = self.fc.bias
        self.fc = self.fc_new

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        feature = output.view(output.size(0), -1)
        output = self.fc(feature)
        return feature, output

class upper_classfication_resnet(nn.Module):

    def __init__(self, block, num_block, init_num_classes=100):
        super().__init__()
        self.init_num_classes = init_num_classes
        self.block_expansion = block.expansion

        self.resnet = ResNet(block, num_block, init_num_classes)
        
        self.fc1 = nn.Linear(512 * self.block_expansion + init_num_classes, self.init_num_classes)

        #self.fc1 = nn.Linear(512 * self.block_expansion + init_num_classes, 512 * self.block_expansion)
        #self.fc2 = nn.Linear(512 * self.block_expansion, 512 * self.block_expansion)
        #self.fc3 = nn.Linear(512 * self.block_expansion, self.init_num_classes)
        self.bn1 = nn.BatchNorm1d(512 * self.block_expansion)
        self.bn2 = nn.BatchNorm1d(self.init_num_classes)

    def _add_class(self, add_num_classed):
        self.fc_new = nn.Linear(512 * self.block_expansion, self.init_num_classes+add_num_classed)
        nn.init.zeros_(self.fc_new.weight)
        nn.init.zeros_(self.fc_new.bias)
        self.fc_new.weight[:self.init_num_classes,:] = self.fc4.weight
        self.fc_new.bias[:self.init_num_classes,:] = self.fc4.bias
        self.fc4 = self.fc_new

    def forward(self, x):
        feature, output1 = self.resnet(x)

        softmax_output = F.softmax(output1)
        cat_input = torch.cat( (feature, softmax_output), 1)

        output2 = self.bn2(self.fc1(cat_input))

        #output = F.relu(self.bn1(self.fc1(cat_input)))
        #output = F.relu(self.bn1(self.fc2(output)))
        #output2 = self.bn2(self.fc3(output))
        return feature, output1, output2




def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet18_upper():
    """ return a ResNet 18 object
    """
    return upper_classfication_resnet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet34_upper():
    """ return a ResNet 34 object
    """
    return upper_classfication_resnet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet50_upper():
    """ return a ResNet 50 object
    """
    return upper_classfication_resnet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
