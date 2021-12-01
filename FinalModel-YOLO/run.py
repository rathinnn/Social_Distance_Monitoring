import cv2
import numpy as np
import time
from calibrateTools import calibrate, transform
from computeDistance import segregateAfterTransform

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

            
            #cv2.circle(frame, ((x + w//2),y + h), radius=2, color=(0, 0, 255), thickness=2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #text = label + ':' + str(p)
            #cv2.putText(frame, text, (x, y+30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            #if writer is None:
            # initialize our video writer
            #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #writer = cv2.VideoWriter("out.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        #Get Classified boxes
        final = segregateAfterTransform(projected, transformation_matrix, distance)
        for bx in final:
            box = bx[0][1]
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])

            #If True then Red
            if(bx[1]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            else:
                #Green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    time_taken = time.time() - starting_time
    sum_time += time_taken


    
    cv2.imshow("Frame", frame)
    #writer.write(frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        #writer.release()
        break

avgProc = sum_time/frame_id
print(avgProc)
cv2.destroyAllWindows()