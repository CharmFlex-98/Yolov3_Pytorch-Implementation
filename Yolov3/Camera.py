import time
from utils import *
from Darknet import Darknet
import cv2


def draw_box(x, frame):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = frame
    cls = classes[int(x[-1])]
    cls_conf = '{0}%'.format(round((x[-2].item() * 100), 1))
    label = "{0}".format(cls)
    cv2.rectangle(img, c1, c2, class_col[cls], 1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    # get the size (width, height) of text
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, class_col[cls], -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255],1)
    conf_size = cv2.getTextSize(cls_conf, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]  # get the size (width, height) of text
    c1 = c1[0], c2[1]
    c2 = c1[0] + conf_size[0] + 3, c1[1] + conf_size[1] + 4
    cv2.rectangle(img, c1, c2, class_col[cls], -1)
    cv2.putText(img, cls_conf, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    # text coordinates count from bottom left corner



if __name__ == '__main__':
    args = arg_parse()
    num_classes = 80
    classes = load_classes('data/coco.names')
    class_col = class_colour(classes)
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    model.net_info['height'] = args.reso
    input_dim = int(model.net_info["height"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = cv2.VideoCapture(0)

    assert camera.isOpened(), 'Cannot run the camera!'
    frames = 0

    model.to(device)
    model.eval()

    start = time.time()
    while camera.isOpened():
        is_record, frame = camera.read()    # the frame is already in nparray form, no need imread again!
        if is_record:
            image, orig_image, orig_dim = prepare_image(frame, input_dim, True)
            image = image.to(device)

            prediction = model(image)
            output = non_max_suppression(prediction, float(args.confidence), num_classes, nms_conf=float(args.nms_thresh))

            if type(output) == int:  # if no detection
                frames += 1
                print('FPS: {}'.format(frames / time.time() - start))
                cv2.imshow('Camera', orig_image)
                key = cv2.waitKey(1)  # wait for user to press any key in 1ms. If press, return ASCII of the key
                if key & 0xFF == ord('q'):  # 0xFF = 1111 1111
                    break
                continue

            if output.shape[0] > 0:  # if there is detection
                output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(input_dim)) / input_dim
                output[:, [1, 3]] *= frame.shape[1]
                output[:, [2, 4]] *= frame.shape[0]

                list(map(lambda x: draw_box(x, orig_image), output))

            cv2.imshow("Camera", orig_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS: {}".format(frames / time.time()))
        else:
            break
