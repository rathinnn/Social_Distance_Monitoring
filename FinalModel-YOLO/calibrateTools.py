import cv2
import numpy as np

#defining function for click event i.e marks a point on image with red
#where left mouse button is clicked
points =[]
coor = []

#Tracking the number of clicks
nClick = 0
img = None

def clickEvent(event, x, y, flags, params):
	
	
	if event == cv2.EVENT_LBUTTONDOWN:
		global nClick
		global img
		#First part of the calibration is for transformation matrix
		if(nClick < 4):
			print('Perspective Click ')
			print(x, ',', y) 
			points.append([x,y])
			center = (x, y) 
			radius = 1 
			cv2.circle(img, center, radius,(0,0,255), 5) 
			cv2.imshow('image', img)
		
		#Secon part will give the distance after transformation
		else:
			print('Distance Click ')
			print(x, ',', y)
			coor.append([x,y])
			center = (x, y)
			radius = 1
			cv2.circle(img, center, radius,(255,0,0), 5)
			cv2.imshow('image', img)
		nClick+=1


def computeParams(img):

	pts1 = np.float32(points)

	#reference points to calculate transform
	transformed_points = [[0,0], [500,0], [0,500], [500,500]]
	pts2 = np.float32(transformed_points)

	#calculate Perspective transform
	transformation_matrix = cv2.getPerspectiveTransform(pts1, pts2)
	transformed = cv2.warpPerspective(img, transformation_matrix, (300,300))


	#Transform the points input for distance
	result = transform(coor, transformation_matrix)

	#Calculate Euclidean distance
	one = result[0]
	two = result[1]
	diff = one  - two
	print('Distance Limit in pixels ')
	distance = np.sqrt(np.dot(diff.T, diff))
	print(distance)

	return [transformation_matrix, transformed, distance, points]

def transform(points, M):
	points = np.float32(points)
	result = cv2.perspectiveTransform(points[None, :, :], M)
	ret = []
	for i in range(0,result.shape[1]):
		ret.append(result[0][i])
	return ret


def calibrate(vid):
	cap = cv2.VideoCapture(vid)
	if(cap.isOpened()):
		ret, frame = cap.read()
	else:
		print('No Frame')
	cap.release()
	global img
	img = frame
	cv2.imshow('image',img)
	cv2.setMouseCallback('image',clickEvent)
	cv2.waitKey(0)
	ret = computeParams(img)
	cv2.destroyAllWindows()
	return ret


if __name__ == "__main__":
	ret = calibrate('test.avi')