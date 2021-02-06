import torch
import numpy as np
import cv2
import random
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


def predict_transform(prediction, input_dim, anchors, num_classes, device):
    batch_size = prediction.shape[0]
    stride = input_dim // prediction.shape[2]
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    prediction = prediction.reshape(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.permute(0, 2, 1).contiguous()
    prediction = prediction.reshape(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.tensor(a).float().reshape(-1, 1)
    y_offset = torch.tensor(b).float().reshape(-1, 1)

    x_offset = x_offset.to(device)
    y_offset = y_offset.to(device)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    anchors = torch.tensor(anchors).float()

    anchors = anchors.to(device)

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction


def bbox_iou(bbox1, bbox2):
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # if either size is negative, return zero
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).to(device)) * torch.max(
        inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).to(device))

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def non_max_suppression(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask  # 0 out the row with lower than confidence

    prediction_temp = prediction.clone().detach()  #####
    prediction_temp[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    prediction_temp[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    prediction_temp[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)  # intermediate would change, so use temp
    prediction_temp[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = prediction_temp[:, :, :4]

    batch_size = prediction.shape[0]

    output = torch.tensor([])
    write = False

    for index in range(batch_size):
        image_tensor = prediction[index]  # 2D TENSOR

        class_max_score, class_index = torch.max(image_tensor[:, 5:5 + num_classes], 1)
        class_max_score = class_max_score.float().unsqueeze(1)
        class_index = class_index.float().unsqueeze(1)

        new_row = (image_tensor[:, :5], class_max_score, class_index)
        image_tensor = torch.cat(new_row, 1)  # image tensor with highest confident score and index position each row

        image_tensor = image_tensor[image_tensor[:, 4] != 0].reshape(-1, 7)

        # if image_tensor.shape[0]==0:
        #     output=0
        #     continue

        # in case if all cells has very low confident score, no object will be detected since
        # it is zero, and they all will be removed.
        try:
            class_index_list = torch.unique((image_tensor[:, -1]))  # collect the classes appear on image
        except:
            continue

        for _class in class_index_list:
            unique_class = image_tensor[image_tensor[:, -1] == _class].reshape(-1, 7)
            sorted_index = torch.sort(unique_class[:, 4], descending=True)[1]
            unique_class = unique_class[sorted_index]
            amount = unique_class.shape[0]

            # Non-MAx Suppression
            for i in range(amount):
                try:
                    # output all the bbox ious for filtering in future
                    ious = bbox_iou(unique_class[i].unsqueeze(0), unique_class[i + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                unique_class[i + 1:] *= iou_mask

                unique_class = unique_class[unique_class[:, 4] != 0].reshape(-1, 7)

            batch_index = unique_class.new_empty(unique_class.shape[0], 1).fill_(index)
            new_row = (batch_index, unique_class)

            if not write:
                output = torch.cat(new_row, 1)
                write = True
            else:
                out = torch.cat(new_row, 1)
                output = torch.cat((output, out))

    if not write:  #########
        return 0
    else:
        return output  # output 2D tensors with detected objects X 8, for the batch with batch_size


def load_classes(class_names):
    with open(class_names) as file:
        names = file.read().split('\n')
        names = [name for name in names if name != '']

        return names


def class_colour(class_names):
    class_col = {}
    for _class in class_names:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        class_col[_class] = [R, G, B]

    return class_col


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    h, w = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    resized_image = torch.tensor(resized_image)

    canvas = torch.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prepare_image(image, input_dim, initialized):
    if not initialized:
        orig_image = cv2.imread(image)
        orig_dim = orig_image.shape[0], orig_image.shape[1]
        # image = letterbox_image(orig_image, (input_dim, input_dim))  # return image with padding to fit input dimension
        image=cv2.resize(orig_image, (input_dim, input_dim))
        image=torch.tensor(image)
        image = torch.flip(image, [2]).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # return with batch size of 1 (4D)
    else:
        orig_image = image
        orig_dim = orig_image.shape[0], orig_image.shape[1]
        image=cv2.resize(orig_image, (input_dim, input_dim))
        image=torch.tensor(image)
        image = torch.flip(image, [2]).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    return image, orig_image, orig_dim
