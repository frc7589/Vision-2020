import cv2
import numpy as np
import pickle
import copy
import subprocess

#finding target contours
def areaFilter(inputContours):
	if inputContours:
		ret = inputContours[0]
		max_area = cv2.contourArea(ret)
		for contour in inputContours:
			area = cv2.contourArea(contour)
			if area > max_area:
				ret = contour
				max_area = area
		return ret
	return np.array([])

def genMask(img):
	#Color filter range
	upper_green = np.array([90,255,255])
	lower_green = np.array([30,70,60])

	hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_img,lower_green,upper_green)
	return mask

# the distant between 2D points
def getDistant(point0,point1):
	xgap = point1[0] - point0[0]
	ygap = point1[1] - point0[1]
	return np.sqrt(xgap*xgap + ygap*ygap)

# the distant between the camera and the object point
def objDistant(point,rvecs,tvecs):
	point = np.matmul(point,rvecs)
	point = np.add(point,tvecs)

	distant = np.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])
	return distant

# remove extreme points and return average of remain points
def avg_points(points):
	for i in range(8):
		for j in range(3):
			avg = [0,0]
			for k in points[i]: 
				avg[0] += k[0]
				avg[1] += k[1]
			avg[0] /= 6-j
			avg[1] /= 6-j

			Maxdis = 0
			del_idx = 0
			for k in range(6-j):
				tmp = getDistant(avg,points[i][k])
				if tmp > Maxdis:
					Maxdis = tmp
					del_idx = k
			del points[i][del_idx]

		avg_points = []
	for i in points:
		tmpx = 0
		tmpy = 0
		for k in i:
			tmpx += k[0]
			tmpy += k[1]
		tmpx /= 3
		tmpy /= 3
		avg_points.append((tmpx,tmpy))

	avg_points = np.array(avg_points,dtype=np.float32)
	return avg_points
	
def main():
	#camera init
	subprocess.call('v4l2-ctl -d /dev/video0 -c exposure_auto=1',shell=True)
	subprocess.call('v4l2-ctl -d /dev/video0 -c exposure_absolute=50',shell=True)
	cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

	#opening cameraMatrix and distCoeffs
	with open('mtx.txt','rb') as file:
		mtx = pickle.load(file)
	with open('dist.txt','rb') as file:
		dist = pickle.load(file)

	#object points(half of hexagon) unit : cm
	longEdge = 24.92375
	shortEdge = 22.38375
	sq3 = np.sqrt(3)
	objp = np.array([ [longEdge*-2,0,0], [shortEdge*-2,0,0], 
					[longEdge*-1,longEdge*sq3*-1,0], [shortEdge*-1,shortEdge*sq3*-1,0],
					[shortEdge,shortEdge*sq3*-1,0], [longEdge,longEdge*sq3*-1,0],
					[shortEdge*2,0,0], [longEdge*2,0,0] ],np.float32)

	#criteria for subpix
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	#other init
	cornerCnt = 0
	corners = [[],[],[],[],[],[],[],[]]

	while True:
		#catching frame
		_, frame = cap.read()
		cv2.imshow('live',cv2.resize(frame,(1024,540)))

		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		mask = genMask(frame)

		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
		target_cont = areaFilter(contours)
		
		if target_cont.any() :
			epsilon = 0.0095*cv2.arcLength(target_cont,True)
			approx = cv2.approxPolyDP(target_cont,epsilon,True)
			
			ret = []
			for i in approx:
				for j in i:
					ret.append(j)
			approx = np.array(ret,dtype=np.float32)

			if len(approx) == 8:
				subpix = cv2.cornerSubPix(gray,approx,(20,20),(-1,-1),criteria)
				order = np.lexsort((subpix[:,1],subpix[:,0]))
				subpix = subpix[order]

				subpix = subpix.tolist()

				if len(subpix) == 8:
					for i in range(8):
						corners[i].append(subpix[i])
					cornerCnt += 1

					if cornerCnt == 6:	
						clone_corners = copy.deepcopy(corners)
						avg_corners = avg_points(clone_corners)

						_,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp,avg_corners, mtx, dist)
						distant = objDistant([0,0,0],rvecs,tvecs)
						
						for i in corners:
							del i[0]
						cornerCnt -= 1

						print(distant)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()