from collections import OrderedDict
import time

import numpy as np
from openvino.inference_engine import IECore
from scipy.spatial import distance

from utils import *


def predict(frame, inference_request):
    """
    Predict bounding boxes and confidence scores for vehicles
    :param frame: video frame
    :param inference_request: an executable network on CPU device with synchronous execution
    :return: predictions with confidence score greater than some threshold
    """
    # construct an input blob, run and extract the output blob

    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    input_blob = next(iter(inference_request.inputs))
    inference_request.infer(inputs={input_blob: blob})
    output_blob_name = next(iter(inference_request.output_blobs))
    output = inference_request.output_blobs[output_blob_name].buffer
    out = np.array(output)

    # output blob has a shape [1, 1, N, 7], where N is the number of detected bounding boxes
    # go through all detections and leave only the ones with a sufficient confidence score

    predictions = []
    for detection in out.reshape(-1, 7):
        image_id, label, conf, x_min, y_min, x_max, y_max = detection
        if conf > 0.3:
            boxes = ((x_min, y_min), (x_max, y_max))
            prediction = (conf, boxes)
            predictions.append(prediction)

    return predictions


class Tracker():
    """
    implementation of a multiple object tracker
    """

    def __init__(self):
        """
        id_counter: unique object id
        objects: centers of current positions of objects
        previous: centers of previous objects positions
        lost: saves the number of frames during which an object is lost and not detected
        lost_tolerance: what is the maximum tolerable number of such frames
        colors: saves different colors for tracks
        """
        self.id_counter = 0
        self.objects = OrderedDict()
        self.previous = OrderedDict()
        self.lost = OrderedDict()
        self.lost_tolerance = 10
        self.colors = OrderedDict()

    def add_object(self, centroid):
        """
        add new object with unique id equal to the current id_counter
        """
        self.objects[self.id_counter] = centroid
        self.previous[self.id_counter] = centroid
        self.lost[self.id_counter] = 0
        self.id_counter += 1
        self.colors[self.id_counter] = get_color(self.id_counter)

    def remove_object(self, id):
        """
        remove object info when it is lost
        """
        del self.objects[id]
        del self.previous[id]
        del self.lost[id]

    def update(self, boxes):
        """
        update tracking info with newly found bounding boxes
        """
        # if there are no bounding boxes on the current frame, don't update anything, except for the case
        # when a tolerable loss number is exceeded
        if len(boxes) == 0:
            for id in list(self.lost.keys()):
                self.lost[id] += 1
                if self.lost[id] > self.lost_tolerance:
                    self.remove_object(id)
            return self.objects, self.previous, self.colors

        # new_centroids - centers of new bounding boxes
        new_centroids = np.zeros((len(boxes), 2), dtype="int")
        for (i, (min_x, min_y, max_x, max_y)) in enumerate(boxes):
            # get the centroid coordinates
            x = int((min_x + max_x) / 2.0)
            y = int((min_y + max_y) / 2.0)
            new_centroids[i] = (x, y)

        # match a new input centroid to an existing object centroid using Euclidean distance
        # update current centroids and save the previous ones
        if len(self.objects) == 0:
            for i in range(0, len(new_centroids)):
                self.add_object(new_centroids[i])
        else:
            ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            dist = distance.cdist(np.array(object_centroids), new_centroids)
            rows = dist.min(axis=1).argsort()
            cols = dist.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols or dist[row][col] > 150:
                    continue
                object_id = ids[row]
                self.previous[object_id] = self.objects[object_id]
                self.objects[object_id] = new_centroids[col]
                self.lost[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dist.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dist.shape[1])).difference(used_cols)
            if dist.shape[0] >= dist.shape[1]:
                for row in unused_rows:
                    object_id = ids[row]
                    self.lost[object_id] += 1
                    if self.lost[object_id] > self.lost_tolerance:
                        self.remove_object(object_id)
            else:
                for col in unused_cols:
                    self.add_object(new_centroids[col])
        return self.objects, self.previous, self.colors


def main():
    args = parse_args()

    cap, meta = get_video_capture(args.video)
    out = get_video_writer(args.output, meta)

    width = meta['width']
    height = meta['height']

    # load the downloaded model structure (xml file) (pretrained openvinno vehicle-detection nn)
    # and the model weights (bin file) into the network variable
    # build the executable network
    ie = IECore()
    network = ie.read_network(model=args.xml_path, weights=args.bin_path)
    executable_network = ie.load_network(network, device_name='CPU')
    inference_request = executable_network.requests[0]

    # tracker initialization
    tracker = Tracker()
    points = {}

    start_time = time.time()
    frames = 1

    ret, frame = cap.read()
    while frame is not None:
        image = frame.copy()
        # get predictions
        predictions = predict(frame, inference_request)
        boxes = []

        for (i, pred) in enumerate(predictions):
            # extract prediction data
            (pred_conf, pred_boxpts) = pred
            ((x_min, y_min), (x_max, y_max)) = pred_boxpts

            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            # draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                          (0, 0, 255), 1)
            boxes.append((x_min, y_min, x_max, y_max))

        # update tracker and get new centroids
        objects, previous, colors = tracker.update(boxes)

        # save all previous positions for each track and draw them as long as the track is valid
        for (i, centroid) in objects.items():
            current_centroid = objects.get(i)
            previous_centroid = previous.get(i)
            color = colors.get(i)

            if i in points:
                points[i].append((previous_centroid[0], previous_centroid[1]))
            else:
                points[i] = [(previous_centroid[0], previous_centroid[1])]

            cv2.circle(image, (current_centroid[0], current_centroid[1]), 2, [0, 255, 0], -1)
            cv2.line(image, (previous_centroid[0], previous_centroid[1]), (current_centroid[0], current_centroid[1]),
                     [0, 255, 0], 2)

            if len(points[i]) > 1:
                draw_track(image, points[i], [0, 255, 0])

        # write new frame
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out.write(frame)
        ret, frame = cap.read()
        frames += 1

    end_time = time.time()
    seconds = end_time - start_time
    fps = frames / seconds

    print('processing time: ', "{:.2f}".format(seconds))
    print('processing fps: ', "{:.2f}".format(fps))
    out.release()


if __name__ == '__main__':
    main()
