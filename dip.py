import cv2
import numpy as np
import pyautogui
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


hand_hist = None
def rgb_to_hsv(r, g, b):
	r, g, b = r/255.0, g/255.0, b/255.0
	mx = max(r, g, b)
	mn = min(r, g, b)
	df = mx-mn
	if mx == mn:
		h = 0
	elif mx == r:
		h = (60 * ((g-b)/df) + 360) % 360
	elif mx == g:
		h = (60 * ((b-r)/df) + 120) % 360
	elif mx == b:
		h = (60 * ((r-g)/df) + 240) % 360
	if mx == 0:
		s = 0
	else:
		s = (df/mx)*100
	v = mx*100
	return h, s, v

def dominant_color(frame):
	reshaped_frame = frame.reshape((frame.shape[0] * frame.shape[1], 3))
	clt = KMeans(n_clusters = 3)
	clt.fit(reshaped_frame)
	average = frame.mean(axis=0).mean(axis=0)
	ans = closest(clt.cluster_centers_, average)
	H, S, V = rgb_to_hsv(ans[2], ans[1], ans[0])
	return np.array([H/2, S*2.55, V*2.55])
	 

def closest(clts, avg):
	diff = 1000
	ans = clts[0]
	for i in list(clts):
		i = list(i)
		curdiff = abs(i[0] - avg[0]) + abs(i[1] - avg[1]) + abs(i[2] - avg[2])
		if curdiff < diff:
			diff = curdiff
			ans = i
	avg = np.array(avg)
	return np.array(ans)

def blockFace(image):
	global im, faceCascade
	
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	h_r,w_r,ch = image.shape
	roi = np.zeros((h_r,w_r,3), np.uint8)
	roi[:,:] = (255,255,255)
	faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
	faces = faceCascade.detectMultiScale(gray, 1.3, 4)
	for (x,y,w,h) in faces:
		w1 = int(w*0.10)
		w2 = int(w*0.99)
		h1 = int(h*0.01)
		h2 = int(h*0.99)
		box = cv2.rectangle(image,(x+w1,y+h1),(x+w2,y+h2),(0,0,0),2)
		roi = image[y+h1:y+h2, x+w1:x+w2]
		arr = np.ones(image.shape)
		image[y+h1:y+h2 , x+w1:x+w2 , :] = 0


capture = cv2.VideoCapture(0)
dom_color = np.array( [  6.76466883 , 65.15396167 ,109.86372197])
var = 0
while capture.isOpened():
	var += 1
	pressed_key = cv2.waitKey(1)
	_, frame = capture.read()
	blockFace(frame)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	if len(dom_color) != 0:
		mask = cv2.inRange(hsv, dom_color - 10, dom_color + 10)
		result = cv2.bitwise_and(frame,frame, mask= mask)
	try:
		kernel = np.ones((8, 8), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations=8)
		mask = cv2.erode(mask, kernel, iterations=4)
		cv2.imshow("Mask", mask)
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt=max(contours, key=lambda x: cv2.contourArea(x))
		center = cv2.moments(cnt)
		a = int(center["m10"] / center["m00"])
		b = int(center["m01"] / center["m00"])
		print(a,b)
		cv2.circle(frame, (a, b), 10, (255, 0, 0), -1)

		cv2.imshow("Frame",frame)
		if np.sum(mask) > 3*10**6 and var%10 == 0:
			pyautogui.hotkey('space')
			print('space')

	except Exception as e:
		print(str(e))
		cv2.imshow("Live Feed", frame)


	if pressed_key & 0xFF == ord('z'):
		dom_color = dominant_color(frame)
		print('hsv:', dom_color)
		print('hsvActual:', dom_color[0]*3.6, dom_color[1]/2.55, dom_color[2]/2.55)
	if pressed_key == 27:
		break
	
cv2.destroyAllWindows()
capture.release()