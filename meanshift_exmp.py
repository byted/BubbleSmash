import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(5, 1)

cap_width = cap.get(3)
cap_height = cap.get(4)

roi_width = 80
roi_height = 120

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r = int(cap_height / 2 - roi_height / 2)
h = roi_height
c = int(cap_width / 5) # not centered for right hand use
w = roi_width
track_window = (c,r,w,h)

def startTracking(fr, r,h,c,w):
    roi = frame[r:r+h, c:c+w]
    hs_r =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ma = cv2.inRange(hs_r, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    r_h = cv2.calcHist([hs_r],[0],ma,[180],[0,180])
    cv2.normalize(r_h,r_h,0,255,cv2.NORM_MINMAX)

    return r_h

# set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#roi_hist = doit(frame,r,h,c,w)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

tracking = False
roi_hist = None

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # apply meanshift to get the new location
        if tracking:
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            print "dst:",dst
            print "track_window:",track_window
            print "term_crit:",term_crit
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
        else:
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

        # Draw it on image
        img2 = cv2.flip(img2, flipCode=1)
        cv2.imshow('img2',img2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            tracking = True
            roi_hist = startTracking(frame,r,h,c,w)
        elif key & 0xFF == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
