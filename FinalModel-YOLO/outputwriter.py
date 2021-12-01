import cv2
class OutputWriter:
	def __init__(self, filename1, filename2, size):
		self.writer = cv2.VideoWriter(filename1,
								cv2.VideoWriter_fourcc(*'MJPG'),
								30, size, True)

		newSize = list(size)
		newSize[0] = newSize[0] * 2
		newSize = tuple(newSize)

		self.writer2 = cv2.VideoWriter(filename2,
								cv2.VideoWriter_fourcc(*'MJPG'),
								30, newSize, True)

	def writeMerged(self, org, out):
		final = cv2.hconcat([org, out])
		self.writer2.write(final)

	def write(self, frame, final):
		newFrame = frame.copy()
		for bx in final:
			box = bx[0][1]
			(x, y) = (box[0], box[1])
			(w, h) = (box[2], box[3])

			#If True then Red
			if(bx[1]):
				cv2.rectangle(newFrame, (x, y), (x + w, y + h), (0,0,255), 2)
			else:
				#Green
				cv2.rectangle(newFrame, (x, y), (x + w, y + h), (0,255,0), 2)

		self.writer.write(newFrame)
		self.writeMerged(frame, newFrame)

		return newFrame
