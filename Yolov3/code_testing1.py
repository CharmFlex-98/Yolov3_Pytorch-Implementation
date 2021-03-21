import cv2
import random


class yolov4:
    def __init__(self):
        self.cam = True
        self.image_path = ''
        self.names_path = r'C:\Users\ASUS\darknet\build\darknet\x64\data\coco.names'
        self.cfg_path = r'C:\Users\ASUS\darknet\build\darknet\x64\cfg\yolov4-tiny.cfg'
        self.weights_path = r'C:\Users\ASUS\darknet\build\darknet\x64\weights\yolov4-tiny.weights'
        self.width = 416
        self.height = 416
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.3

        self.person = []
        self.cell_phone = []

        self.class_colour_dict = {}
        with open(self.names_path, 'r') as file:
            self.class_names = file.read().rstrip('\n').split('\n')
            self.class_colour(self.class_names)

        self.model = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def find_object(self, _outputs, _frame):
        bboxes = []
        class_ids = []
        confidences = []
        frame_h, frame_w = _frame.shape[:2]
        for output in _outputs:  # each 3 different scales
            for detection in output:
                scores = detection[5:]
                detected_classID = scores.argmax(0)
                confidence = scores[detected_classID]
                if confidence > self.confidence_threshold:
                    w, h = int(detection[2] * frame_w), int(detection[3] * frame_h)
                    x, y = int(detection[0] * frame_w - w / 2), int(
                        detection[1] * frame_h - h / 2)  # upper left coordinate
                    bboxes.append([x, y, w, h])
                    class_ids.append(detected_classID)
                    confidences.append(float(confidence))

        object_index = cv2.dnn.NMSBoxes(bboxes, confidences, self.confidence_threshold, self.nms_threshold)
        for i in object_index:
            i = i[0]
            box = bboxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            self.collect_true_sample(self.class_names[class_ids[i]], [x, y, w, h])
            if self.class_names[class_ids[i]] == 'person':
                continue
            cv2.rectangle(_frame, (x, y), (x + w, y + h), self.class_colour_dict[self.class_names[class_ids[i]]], 2)
            cv2.putText(_frame, '{} : {}'.format(self.class_names[class_ids[i]], round(confidences[i] * 100, 1)),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, self.class_colour_dict[self.class_names[class_ids[i]]], 2)

    def collect_true_sample(self, class_name, coordinate):
        if class_name == 'person':
            self.person.append(coordinate)
        elif class_name == 'cell phone':
            self.cell_phone.append(coordinate)
        else:
            pass

    def checking(self, _frame):
        print(len(self.person))
        for p in self.person:
            equipped = False
            for index, item in enumerate(self.cell_phone):
                if p[0] <= item[0] + item[2] / 2 <= p[0] + p[2] and p[1] <= item[1] + item[3] / 2 <= p[1] + p[3]:
                    cv2.rectangle(_frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), [255, 0, 255], 2)
                    cv2.putText(_frame, '{}'.format('good!'), (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                                [255, 0, 255])
                    equipped = True
                    break
            if not equipped:
                cv2.rectangle(_frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), [255, 255, 255], 2)
                cv2.putText(_frame, '{}'.format('bad!'), (p[0], p[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            [255, 255, 255])

    def class_colour(self, classes):
        for _class in classes:
            R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            colour = (R, G, B)
            self.class_colour_dict[_class] = colour

    def detection(self):
        if self.cam:
            camera = cv2.VideoCapture(0)
            camera.set(3, self.width)
            camera.set(4, self.height)

            while True:
                success, frame = camera.read()
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (self.width, self.height), [0, 0, 0], crop=False)
                self.model.setInput(blob)

                layers = self.model.getLayerNames()
                outputNames = [layers[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
                outputs = self.model.forward(outputNames)  # these are the outputs with results

                self.find_object(outputs, frame)
                self.checking(frame)

                cv2.imshow('cam', frame)
                cv2.waitKey(1)

        else:
            image = cv2.imread(self.image_path)
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.width, self.height), [0, 0, 0], crop=False)
            self.model.setInput(blob)

            layers = self.model.getLayerNames()
            outputNames = [layers[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
            outputs = self.model.forward(outputNames)  # these are the outputs with results
            self.find_object(outputs, image)
            cv2.imwrite('opencv2.jpg', image)


run = yolov4()
run.detection()
