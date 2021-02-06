import time
import torch
from utils import *
import os
from Darknet import Darknet

def draw_box(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = classes[int(x[-1])]
    cls_conf = '{0}%'.format(round((x[-2].item() * 100), 1))
    label = "{0}".format(cls)
    cv2.rectangle(img, c1, c2, class_col[cls], 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]  # get the size (width, height) of text
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, class_col[cls], -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    conf_size = cv2.getTextSize(cls_conf, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]  # get the size (width, height) of text
    c1 = c1[0], c2[1]
    c2 = c1[0] + conf_size[0] + 3, c1[1] + conf_size[1] + 4
    cv2.rectangle(img, c1, c2, class_col[cls], -1)
    cv2.putText(img, cls_conf, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    # text coordinates count from bottom left corner


if __name__ == '__main__':
    # initialize hyper-parameters
    # ---------------------------------------------------------=

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    num_classes = 80
    classes = load_classes('data/coco.names')
    class_col = class_colour(classes)

    # -----------------------------------------------------------

    start = 0
    print('Loading network...')
    model = Darknet(args.cfgfile)  # create modules
    model.load_weights(args.weightsfile)
    print('Network loaded!')

    model.net_info['height'] = args.reso
    input_dim = int(model.net_info['height'])
    assert input_dim % 32 == 0
    assert input_dim > 32

    model.to(device)
    model.eval()

    read_dir = time.time()
    try:
        images_list = [os.path.join(os.getcwd(), images, img) for img in os.listdir(images) if
                       (img.endswith('jpg') or img.endswith('jpeg') or img.endswith('png'))]
    except NotADirectoryError:
        print('Please make a file containing images for detecting')
    except FileNotFoundError:
        print('No such image or directory with the name {}'.format('\'' + images + '\''))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    # return (image (4D), orig_image, input_dim)
    batches = list(map(lambda x, y: prepare_image(x, y, False), images_list,
                       [input_dim for i in range(len(images_list))]))
    image_batches = [i[0] for i in batches]  # (4D_image, 4D_image, ...)
    orig_image_list = [i[1] for i in batches]
    orig_dim_list = [i[2] for i in batches]
    orig_dim_list = torch.tensor(orig_dim_list).float().repeat(1, 2).to(device)  # 2D tensor

    leftover = 0

    if len(images_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = int(len(images_list) / batch_size + 1)
        # [B X 3D image, B X 3D image, B X 3D image, ...]
        image_batches = [torch.cat((image_batches[i * batch_size: min((i + 1) * batch_size),
                                    len(image_batches)])) for i in range(num_batches)]

    write = 0
    start_det_loop = time.time()
    for i, images in enumerate(image_batches):   # for x in [B X 3D image, B X 3D image, B X 3D image, ...]
        start = time.time()
        images = images.to(device)

        with torch.no_grad():
            # return (batch size X grid*grid*3 X 85) with 3 different scales concatenate along 1 axis
            prediction = model(images)

        # perform filtering and non_max suppression
        # output (num_detections X 8) for each batch, concatenate in dim 0
        # output 2D tensors with (detected objects X 8), for the batch with batch_size
        # if a batch has 8 images, the output has all detections in 2D from this 8 images.
        prediction = non_max_suppression(prediction, confidence, num_classes, nms_conf=nms_thresh)

        if type(prediction) == int:  # no detection at all in the batch  #####
            continue

        end = time.time()

        if len(prediction) != 0:
            prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            # output 2D tensors with (detected objects X 8), all images (all batches)
            output = torch.cat((output, prediction))

        for index, image in enumerate(images_list[i * batch_size:min((i + 1) * batch_size, len(images_list))]):
            image_id = i * batch_size + index
            # output all objects detected in an image
            objects = [classes[int(object[-1])] for object in output if int(object[0]) == image_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objects)))
            print("----------------------------------------------------------")

        if device == 'cuda':
            torch.cuda.synchronize()

    try:
        output
    except NameError:  # no detection in all images
        print('No detections were made')
        exit()

    orig_dim_list = torch.index_select(orig_dim_list, 0, output[:, 0].long())
    # scaling_factor = torch.min(input_dim / orig_dim_list, 1)[0].reshape(-1, 1)  #######
    #
    # output[:, [1, 3]] -= (-scaling_factor * orig_dim_list[:, 1].reshape(-1, 1) + input_dim) / 2
    # output[:, [2, 4]] -= (-scaling_factor * orig_dim_list[:, 0].reshape(-1, 1) + input_dim) / 2
    #
    # output[:, 1:5] /= scaling_factor
    output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, input_dim) / input_dim
    output[:, [1, 3]] *= orig_dim_list[:, [1]]
    output[:, [2, 4]] *= orig_dim_list[:, [0]]

    for i in range(output.shape[0]):  # for i in all images
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, orig_dim_list[i, 1])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, orig_dim_list[i, 0])

    output_recast = time.time()

    class_load = time.time()

    draw = time.time()

    list(map(lambda x: draw_box(x, orig_image_list), output))

    det_names = [args.det + '\\' + 'detection_' + file.split('\\')[-1] for file in images_list]

    list(map(cv2.imwrite, det_names, orig_image_list))

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(images_list)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(images_list)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
