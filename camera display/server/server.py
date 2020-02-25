from networktables import NetworkTables
import os
import netifaces as ni
import numpy as np

stopCap = False
cap = []
capId = 0

def changes():
	global capId
	preCheck = False
	while not stopCap:
		check = vt.getBoolean('start button',False)
		if check and not preCheck:
			capId = (capId+1)%2	
		preCheck = check
	return

def openCam(cam_id):
	cap = cv2.VideoCapture(cam_id,cv2.CAP_V4L2)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH ,640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 30)

	return cap


def main():
	global cap,stopCap,vt

	roborio_ip = '10.75.89.2'
	NWTable = 'vision table'

	cap.append(openCam(0))
	cap.append(openCam(1))

	NetworkTables.initialize(server=roborio_ip)
	vt = NetworkTables.getTable(NWTable)
	
	threading.Thread(target=changes).start()

	codec = cv2.VideoWriter_fourcc(*'av01')
	pipeline = "appsrc ! autovideoconvert ! omxvp9enc bitrate=1000000 control-rate=3 ! gdppay ! tcpserversink host=0.0.0.0 port=1180 sync=false async=false"

	vw = cv2.VideoWriter(pipeline,cv2.CAP_GSTREAMER,codec,30.0,(640,480))
	
	while True:
		ret,frame = cap[capId].read()
		#gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		vw.write(frame)
	
			
if __name__ == '__main__':
	main()
