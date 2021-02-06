import torch
import numpy as np
import torch.nn as nn
from utils import *
import cv2
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as file:
        lines = file.read().split('\n')
        lines = [line for line in lines if (len(line) > 0 and line[0] is not '#')]
        lines = [line.lstrip().rstrip() for line in lines]

        block = {}
        blocks = []
        for line in lines:
            if line[0] == '[':
                if block:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()

        blocks.append(block)

        return blocks


def create_module(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    layer_index = 0
    prev_filters = 3
    output_filters = []

    for block in blocks:
        module = nn.Sequential()

        if block['type'] == 'net':
            continue

        if block['type'] == 'convolutional':
            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filter_size = int(block['size'])
            stride = int(block['stride'])
            pad = int(block['pad'])
            filters = int(block['filters'])
            activation = block['activation']

            if pad:
                pad = (filter_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, filter_size, stride, pad, bias=bias)
            module.add_module('conv_{}'.format(layer_index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('bn_{}'.format(layer_index), bn)

            if activation == 'leaky':
                act_func = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('act_func_{}'.format(layer_index), act_func)

        elif block['type'] == 'upsample':
            stride = block['stride']
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{}'.format(layer_index), upsample)

        elif block['type'] == 'route':
            block['layers'] = block['layers'].split(',')

            start = int(block['layers'][0])
            if start > 0:
                start = start - layer_index

            try:
                end = int(block['layers'][1])
                if end > 0:
                    end = end - layer_index
            except:
                end = 0

            if end < 0:
                filters = output_filters[layer_index + start] + output_filters[layer_index + end]
            else:
                filters = output_filters[layer_index + start]

            route = EmptyLayer()
            module.add_module('route_{}'.format(layer_index), route)

        elif block['type'] == 'shortcut':
            from_ = int(block['from'])
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(layer_index), shortcut)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{}'.format(layer_index), detection)
        else:
            print('module not found')
            assert False

        print(module, '\n')
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        layer_index += 1

    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_module(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])  # initialize
        self.seen = 0  # initialize

    def get_block(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i in range(len(modules)):

            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample":

                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)  # combine feature map from two different outputs along 1
                outputs[i] = x  # store all output features map in each layer

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_]  # add feature map from previous layer and nth previous layer
                outputs[i] = x

            elif module_type == 'yolo':
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(modules[i]["classes"])

                # Output the result
                # return prediction with batch size X grid*grid*3 X 85
                x = self.module_list[i][0](x, inp_dim, num_classes)

                # if type(x) == int:
                #     continue

                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]

        try:
            return detections  # return (batch size X grid*grid*3 X 85) with 3 different scales concatenate along 1 axis
        except:
            return 0

    def load_weights(self, weightfile):
        print('Loading weights...')

        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, prediction, input_dim, num_classes):
        prediction = prediction.detach()
        prediction = predict_transform(prediction, input_dim, self.anchors, num_classes, device)

        return prediction

