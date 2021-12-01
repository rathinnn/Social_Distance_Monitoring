
import numpy as np
import tensorflow as tf
import cv2
import time
from bird_eye_transform import calibrate, transform
import itertools
import math

class Model:

	def __init__(self, model_path):
		# Initilaize graph object
		self.detection_graph = tf.Graph()
		# Load the model 
		with self.detection_graph.as_default():
			od_graph_def = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(model_path, 'rb') as file:
				serialized_graph = file.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		# Create tf session using graph
		self.sess = tf.compat.v1.Session(graph=self.detection_graph)

	def predict(self,img):
		# Mode requires shape [1, None, None, 3]
		img_exp = np.expand_dims(img, axis=0)
		(boxes, scores, classes) = self.sess.run([self.detection_graph.get_tensor_by_name('detection_boxes:0'), self.detection_graph.get_tensor_by_name('detection_scores:0'), self.detection_graph.get_tensor_by_name('detection_classes:0')],feed_dict={self.detection_graph.get_tensor_by_name('image_tensor:0'): img_exp})
		return (boxes, scores, classes)  


def filter(boxes,scores,classes,height,width):
	 
	#Get coordinates of Only Person class
	#boxes : bounding box coordinates
	#scores : probability between 0 and 1 used to filter weak predictions
	#classes : 1 is human
	#height
	#width 
	array_boxes = list()
	for i in range(boxes.shape[1]):
		# Score needs to be greater than 60 percent
		if int(classes[i]) == 1 and scores[i] > 0.6:
			# Multiply the X coordonnate by the height of the image and the Y coordonate by the width
			# To transform the box value into pixel coordonate values.
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes

def segregateAfterTransform(points, M, distance):

	#First Transforms the points into top down view
	#Appends boolean value of true or false by brute forcing every combination of detection
	#True is not observing social distancing
	#False is observing social distanceing

	#points: points along with bounding box the points are the centroid projected to the ground
	#M Transformation matrix
	#Distance: distance threshold after transformation
	act = []
	all = []
	for point in points:
		#By default initalized to false
		all.append([point, False])
		act.append(point[0])
	
	
	act = np.float32(act)\
	#Calculate Perspective Transform
	result = cv2.perspectiveTransform(act[None, :, :], M)
	transformed = []
	for i in range(0,result.shape[1]):
		transformed.append(result[0][i])

	#Brute force all combinations
	pairs = list(itertools.combinations(range(len(transformed)), 2))
	for i,vec in enumerate(itertools.combinations(transformed, r=2)):
		if math.sqrt( (vec[0][0] - vec[1][0])**2 + (vec[0][1] - vec[1][1])**2 ) < distance:
			first = pairs[i][0]
			second = pairs[i][1]

			#Set to true
			all[first][1] = True
			all[second][1] = True
	
	return all

#First calibrate camera and return required parameters
[transformation_matrix, transformed, distance, points] = calibrate("../test.avi")

#Initalize model using precalculated weights
model_path = "faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"
model = Model(model_path)

#Start Video Capture
video_path = "../test.avi"
vs = cv2.VideoCapture(video_path)
width = vs.get(cv2.CAP_PROP_FRAME_WIDTH )
i = 0

#For Comparing processing time of different models
starting_time = time.time()
frame_id = 0
sum_time = 0
while True:	
	(frame_exists, frame) = vs.read()
	frame_id += 1
	starting_time = time.time()
	#Break if end of video
	if not frame_exists:
		break
	else:
		#Prediction step returns bounding boxes along with all classes detected
		(boxes, scores, classes) =  model.predict(frame)
		
		#Filter out human also at the same time filter out weak predictions
		array_boxes_detected = filter(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])

		#Project the centroid to the ground
		projected = []
		for index, box in enumerate(array_boxes_detected):
			(x, y) = (box[1], box[0])
			(x2, y2) = (box[3], box[2])
			projected.append( [[((x + x2)//2),y2], box]  )
			cv2.circle(frame, (((x + x2)//2),y2), radius=4, color=(0, 0, 255), thickness=6)

		#Necessary to avoid empty predictions for weak classifiers
		if(len(projected) > 0):
			final = segregateAfterTransform(projected, transformation_matrix, distance)
		else:
			final = []
		for bx in final:
			box = bx[0][1]
			(x, y) = (box[1], box[0])
			(x2, y2) = (box[3], box[2])
			if(bx[1]):
				cv2.rectangle(frame, (x, y), (x2, y2), (0,0,255), 2)
			else:
				cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 2)

	time_taken = time.time() - starting_time
	sum_time += time_taken

	cv2.imshow("Output", frame)

	if cv2.waitKey(1) == 27:
			vs.release()
			break

avgProc = sum_time/frame_id
print(avgProc)






