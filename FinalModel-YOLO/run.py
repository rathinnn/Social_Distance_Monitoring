import cv2
import numpy as np
import time
from calibrateTools import calibrate, transform
from computeDistance import segregateAfterTransform
from outputwriter import OutputWriter

##First calibrate camera and return required parameters
[transformation_matrix, transformed, distance, points] = calibrate("test.avi")

#Read saved weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
layer_names =[]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#load input video stream
cap = cv2.VideoCapture("test.avi")

#instantiate a variable 'p' to keep count of persons
p = 0

#For Comparing processing time of different models
starting_time = time.time()
frame_id = 0
sum_time = 0

writer = OutputWriter("out.avi", "merged.avi", (768, 576))
(W, H) = (None, None)
while True:
    ret , frame= cap.read()
    frame_id += 1
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    starting_time = time.time()
    if(not ret):
        break
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    class_ids = []

    # loop over each of the layer outputs
    for out in outs:

        # loop over each of the detections
        for detection in out:

            # extract the class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter out weak predictions
            if confidence > 0.7:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)

                #Dimension of bounding box
                w = int(detection[2] * W)
                h = int(detection[3] * H)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    #detecting persons
    projected = []
    if len(indexes) > 0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label = str(classes[class_ids[i]])
            if label == 'person':
                p=p+1
            else:
                continue

            #Project centroid to ground
            projected.append( [[x + w//2,y + h], boxes[i]])


        #Get Classified boxes
        final = segregateAfterTransform(projected, transformation_matrix, distance)

        frame = writer.write(frame, final)

    time_taken = time.time() - starting_time
    sum_time += time_taken



    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        break

avgProc = sum_time/frame_id
print(avgProc)
cv2.destroyAllWindows()
