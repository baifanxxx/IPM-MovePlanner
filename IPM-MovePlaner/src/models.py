import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, N_action):
        super(ActorCriticNet, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(51, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.ConvOutSize = self.get_conv_out_size()

        self.Lstm = nn.LSTMCell(self.ConvOutSize * self.ConvOutSize * 128, 512)

        self.Pi = nn.Linear(512, N_action)
        self.V = nn.Linear(512, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_conv_out_size(self):
        test_tensor = torch.FloatTensor(1, 51, 64, 64)
        out_tensor = self.conv4(self.conv3(self.conv2(self.conv1(test_tensor))))
        conv_out_size = out_tensor.size()[-1]
        return conv_out_size

    def forward(self, x, hidden):
        x = x.permute(0, 3, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, self.ConvOutSize * self.ConvOutSize * 128)
        h, c = self.Lstm(x, hidden)

        prob = self.Pi(h)
        prob = F.softmax(prob, dim=-1)

        value = self.V(h)

        return prob, value, (h, c)


class Actor(nn.Module):
    def __init__(self, state_size=[64, 64, 51], hidden_size=512, max_num=25, action_type=5, feature_size=4096):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_space = max_num * action_type
        self.max_num = max_num
        self.action_type = action_type

        self.flat = nn.Flatten()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=state_size[-1],
                               out_channels=64,
                               kernel_size=5,
                               stride=2,
                               padding=2)

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.resconv = nn.Conv2d(in_channels=64,
                               out_channels=256,
                               kernel_size=1,
                               stride=4)
        self.lstm = nn.LSTMCell(feature_size//2, hidden_size)
        self.linear = nn.Linear(hidden_size, self.action_space, bias=False)
        self.fusionlinear = nn.Linear(feature_size + self.action_space + self.max_num, feature_size//2, bias=False)


    def forward(self, x, pre_action, finish_tag, h_0, c_0):
        x = x.float()
        x = x.permute(0, 3, 2, 1)
        c1 = self.lrelu(self.conv1(x))  # 32 x 32 x 64
        res = c1
        c2 = self.lrelu(self.conv2(c1))  # 16 x 16 x 128
        c3 = self.lrelu(self.conv3(c2))  # 8  x  8 x 256
        c3 = c3 + self.lrelu(self.resconv(res))
        c4 = self.conv4(c3)  # 4  x  4 x 256
        flat = self.flat(c4)  # 4096
        feature = torch.cat([flat, pre_action, finish_tag], -1)
        feature = self.fusionlinear(feature)
        h_1, c_1 = self.lstm(feature, (h_0, c_0))
        logits = self.linear(h_1)
        return logits, h_1, c_1


class Critic(nn.Module):
    def __init__(self, hidden_size=512):
        super(Critic, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        x = self.lrelu(self.linear1(x))
        # res = self.linear2(x).squeeze()
        res = self.linear2(x)

        return res



# class Actor(nn.Module):
#     def __init__(self, state_size=[64, 64, 51], hidden_size=512, max_num=25, action_type=5, feature_size=16384):
#         super(Actor, self).__init__()
#         self.state_size = state_size
#         self.action_space = max_num * action_type
#         self.max_num = max_num
#         self.action_type = action_type
#
#         self.flat = nn.Flatten()
#         self.lrelu = nn.LeakyReLU()
#         self.sfmax = nn.Softmax(dim=-1)
#         self.conv1 = nn.Conv2d(in_channels=state_size[-1],
#                                out_channels=64,
#                                kernel_size=5,
#                                stride=2,
#                                padding=2)
#
#         self.conv2 = nn.Conv2d(in_channels=64,
#                                out_channels=128,
#                                kernel_size=3,
#                                stride=2,
#                                padding=1)
#
#         self.conv3 = nn.Conv2d(in_channels=128,
#                                out_channels=128,
#                                kernel_size=3,
#                                stride=1,
#                                padding=1)
#
#         self.conv4 = nn.Conv2d(in_channels=128,
#                                out_channels=256,
#                                kernel_size=3,
#                                stride=2,
#                                padding=1)
#
#         self.resconv1 = nn.Conv2d(in_channels=64,
#                                out_channels=128,
#                                kernel_size=1,
#                                stride=2)
#
#         self.se1 = SELayer(128, 16)
#
#         self.lstm = nn.LSTMCell(4096, hidden_size)
#         self.linear1 = nn.Linear(feature_size + self.action_space + self.max_num, 4096, bias=False)
#         self.linear2 = nn.Linear(hidden_size, int(hidden_size/4), bias=False)
#         self.linear3 = nn.Linear(int(hidden_size/4), self.action_space, bias=False)
#
#
#     def forward(self, x, pre_action, finish_tag, h_0, c_0):
#         x = x.float()
#         x = x.permute(0, 3, 2, 1)
#         c1 = self.lrelu(self.conv1(x))  # 32 x 32 x 64
#
#         res = c1
#         c2 = self.lrelu(self.conv2(c1))  # 16 x 16 x 128
#         c3 = self.lrelu(self.conv3(c2))  # 8  x  8 x 256
#
#         c3 = self.se1(c3)
#         c3 = self.lrelu(c3 + self.resconv1(res))
#         c4 = self.conv4(c3)  # 4  x  4 x 256
#
#         flat = self.flat(c4)  # 16384
#
#         feature = torch.cat([flat, pre_action, finish_tag], -1)
#         feature = self.linear1(feature)
#         h_1, c_1 = self.lstm(feature, (h_0, c_0))
#         h_2 = self.linear2(h_1)
#         logits = self.sfmax(self.linear3(h_2))
#
#         return logits, h_1, c_1
#
#
# class Critic(nn.Module):
#     def __init__(self, hidden_size=512):
#         super(Critic, self).__init__()
#         self.lrelu = nn.LeakyReLU()
#         self.linear1 = nn.Linear(hidden_size, int(hidden_size/4), bias=False)
#         self.linear2 = nn.Linear(int(hidden_size/4), 1, bias=False)
#
#     def forward(self, x):
#         x = self.lrelu(self.linear1(x))
#         v = self.linear2(x)
#         return v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)