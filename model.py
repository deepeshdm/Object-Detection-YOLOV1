
"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.

==> MODEL SUMMARY :
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           9,408
       BatchNorm2d-2         [-1, 64, 224, 224]             128
         LeakyReLU-3         [-1, 64, 224, 224]               0
          CNNBlock-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 192, 112, 112]         110,592
       BatchNorm2d-7        [-1, 192, 112, 112]             384
         LeakyReLU-8        [-1, 192, 112, 112]               0
          CNNBlock-9        [-1, 192, 112, 112]               0
        MaxPool2d-10          [-1, 192, 56, 56]               0
           Conv2d-11          [-1, 128, 56, 56]          24,576
      BatchNorm2d-12          [-1, 128, 56, 56]             256
        LeakyReLU-13          [-1, 128, 56, 56]               0
         CNNBlock-14          [-1, 128, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         294,912
      BatchNorm2d-16          [-1, 256, 56, 56]             512
        LeakyReLU-17          [-1, 256, 56, 56]               0
         CNNBlock-18          [-1, 256, 56, 56]               0
           Conv2d-19          [-1, 256, 56, 56]          65,536
      BatchNorm2d-20          [-1, 256, 56, 56]             512
        LeakyReLU-21          [-1, 256, 56, 56]               0
         CNNBlock-22          [-1, 256, 56, 56]               0
           Conv2d-23          [-1, 512, 56, 56]       1,179,648
      BatchNorm2d-24          [-1, 512, 56, 56]           1,024
        LeakyReLU-25          [-1, 512, 56, 56]               0
         CNNBlock-26          [-1, 512, 56, 56]               0
        MaxPool2d-27          [-1, 512, 28, 28]               0
           Conv2d-28          [-1, 256, 28, 28]         131,072
      BatchNorm2d-29          [-1, 256, 28, 28]             512
        LeakyReLU-30          [-1, 256, 28, 28]               0
         CNNBlock-31          [-1, 256, 28, 28]               0
           Conv2d-32          [-1, 512, 28, 28]       1,179,648
      BatchNorm2d-33          [-1, 512, 28, 28]           1,024
        LeakyReLU-34          [-1, 512, 28, 28]               0
         CNNBlock-35          [-1, 512, 28, 28]               0
           Conv2d-36          [-1, 256, 28, 28]         131,072
      BatchNorm2d-37          [-1, 256, 28, 28]             512
        LeakyReLU-38          [-1, 256, 28, 28]               0
         CNNBlock-39          [-1, 256, 28, 28]               0
           Conv2d-40          [-1, 512, 28, 28]       1,179,648
      BatchNorm2d-41          [-1, 512, 28, 28]           1,024
        LeakyReLU-42          [-1, 512, 28, 28]               0
         CNNBlock-43          [-1, 512, 28, 28]               0
           Conv2d-44          [-1, 256, 28, 28]         131,072
      BatchNorm2d-45          [-1, 256, 28, 28]             512
        LeakyReLU-46          [-1, 256, 28, 28]               0
         CNNBlock-47          [-1, 256, 28, 28]               0
           Conv2d-48          [-1, 512, 28, 28]       1,179,648
      BatchNorm2d-49          [-1, 512, 28, 28]           1,024
        LeakyReLU-50          [-1, 512, 28, 28]               0
         CNNBlock-51          [-1, 512, 28, 28]               0
           Conv2d-52          [-1, 256, 28, 28]         131,072
      BatchNorm2d-53          [-1, 256, 28, 28]             512
        LeakyReLU-54          [-1, 256, 28, 28]               0
         CNNBlock-55          [-1, 256, 28, 28]               0
           Conv2d-56          [-1, 512, 28, 28]       1,179,648
      BatchNorm2d-57          [-1, 512, 28, 28]           1,024
        LeakyReLU-58          [-1, 512, 28, 28]               0
         CNNBlock-59          [-1, 512, 28, 28]               0
           Conv2d-60          [-1, 512, 28, 28]         262,144
      BatchNorm2d-61          [-1, 512, 28, 28]           1,024
        LeakyReLU-62          [-1, 512, 28, 28]               0
         CNNBlock-63          [-1, 512, 28, 28]               0
           Conv2d-64         [-1, 1024, 28, 28]       4,718,592
      BatchNorm2d-65         [-1, 1024, 28, 28]           2,048
        LeakyReLU-66         [-1, 1024, 28, 28]               0
         CNNBlock-67         [-1, 1024, 28, 28]               0
        MaxPool2d-68         [-1, 1024, 14, 14]               0
           Conv2d-69          [-1, 512, 14, 14]         524,288
      BatchNorm2d-70          [-1, 512, 14, 14]           1,024
        LeakyReLU-71          [-1, 512, 14, 14]               0
         CNNBlock-72          [-1, 512, 14, 14]               0
           Conv2d-73         [-1, 1024, 14, 14]       4,718,592
      BatchNorm2d-74         [-1, 1024, 14, 14]           2,048
        LeakyReLU-75         [-1, 1024, 14, 14]               0
         CNNBlock-76         [-1, 1024, 14, 14]               0
           Conv2d-77          [-1, 512, 14, 14]         524,288
      BatchNorm2d-78          [-1, 512, 14, 14]           1,024
        LeakyReLU-79          [-1, 512, 14, 14]               0
         CNNBlock-80          [-1, 512, 14, 14]               0
           Conv2d-81         [-1, 1024, 14, 14]       4,718,592
      BatchNorm2d-82         [-1, 1024, 14, 14]           2,048
        LeakyReLU-83         [-1, 1024, 14, 14]               0
         CNNBlock-84         [-1, 1024, 14, 14]               0
           Conv2d-85         [-1, 1024, 14, 14]       9,437,184
      BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
        LeakyReLU-87         [-1, 1024, 14, 14]               0
         CNNBlock-88         [-1, 1024, 14, 14]               0
           Conv2d-89           [-1, 1024, 7, 7]       9,437,184
      BatchNorm2d-90           [-1, 1024, 7, 7]           2,048
        LeakyReLU-91           [-1, 1024, 7, 7]               0
         CNNBlock-92           [-1, 1024, 7, 7]               0
           Conv2d-93           [-1, 1024, 7, 7]       9,437,184
      BatchNorm2d-94           [-1, 1024, 7, 7]           2,048
        LeakyReLU-95           [-1, 1024, 7, 7]               0
         CNNBlock-96           [-1, 1024, 7, 7]               0
           Conv2d-97           [-1, 1024, 7, 7]       9,437,184
      BatchNorm2d-98           [-1, 1024, 7, 7]           2,048
        LeakyReLU-99           [-1, 1024, 7, 7]               0
        CNNBlock-100           [-1, 1024, 7, 7]               0
         Flatten-101                [-1, 50176]               0
          Linear-102                  [-1, 496]      24,887,792
         Dropout-103                  [-1, 496]               0
       LeakyReLU-104                  [-1, 496]               0
          Linear-105                 [-1, 1470]         730,590
================================================================
Total params: 85,787,534
Trainable params: 85,787,534
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 2.30
Forward/backward pass size (MB): 436.81
Params size (MB): 327.25
Estimated Total Size (MB): 766.36
----------------------------------------------------------------
"""


import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [

    # (kernel_size, filters, stride, padding)

    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


"""
The model takes a batch of Images : (Batch_size,Channels,Height,Width).
Flattened_tensor = S * S * (C + B * 5)
Returns a tensor of shape : (Batch_size,Flattened_tensor).
The Flattened tensor can be reshaped to SxSx(C + B*5) 
to get the output tensor for each image.
"""


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )
