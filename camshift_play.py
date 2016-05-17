import numpy as np
import cv2
import random
import sys

## CIRCLES
class Circle(object):
    def __init__(self, frame_width):
        self.y_pos = -random.randint(0, 100)  # don't let them start all at once
        self.x_pos = random.randint(0, frame_width)
        self.speed = random.randint(5, 25)
        self.visible = True
        self.popped = False
        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.radius = 30

    def down(self, yLimit):
        self.y_pos += self.speed
        ## check if circle moved out of the frame
        if self.y_pos > yLimit:
            self.visible = False

    def draw(self, video_img):
        cv2.circle(video_img, (self.x_pos, self.y_pos), self.radius, self.color, -1)

    def move(self, video_img):
        ## get frame height
        height,_,_ = video_img.shape

        self.down(height)
        if self.visible:
            self.draw(video_img)


NUM_OF_CIRCLES = 1
SCORE = 0
MISSES = 0

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

def startTracking(frame, r,h,c,w):
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    #mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((255.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    ## difference?
    #roi_hist = cv2.calcHist([hsv_roi],[0],mask,[256],[0,256])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist

## maths magic help function
def isLeftOfABLine(p1, a, b):
    A = -(b[1] - a[1])
    B = b[0] - a[0]
    C = -(A * a[0] + B * a[1])
    D = A * p1[0] + B * p1[1] + C

    return (D > 0)

## is p1 to the same side of line a-b as p2?
def sameSide(p1, p2, a, b):
    return isLeftOfABLine(p1, a, b) == isLeftOfABLine(p2, a, b)

## is point in triangle?
def inTriangle(p1, a, b, c):
    ## checks for each line between two of the triangle corners,
    ## if the third triangle-point lies to the same side as p1
    ## -> if yes, p1 is inside it
    return sameSide(p1, a, b, c) and sameSide(p1, b, a, c) and sameSide(p1, c, a, b)

## checks if point p is covered in rectangle 'r'
def pointCovered(p, r):
    ## split rectangle into two arbitrary triangles and check
    ## if circle lies in one of them
    firstTri = inTriangle(p, r[0], r[1], r[2])
    secondTri = inTriangle(p, r[0], r[3], r[2])
    
    return firstTri or secondTri

## generate points in circle for better collision detection
def getCircPoints(center, radius):
    points = []
    cx = center[0]
    cy = center[1]
    radius *= 1.1
    RCONST = 0.75
    ## move right
    points.append((cx+radius, cy))
    ## move down
    points.append((cx, cy+radius))
    ## move left
    points.append((cx-radius, cy))
    ## move up
    points.append((cx, cy-radius))
    ## move right down
    points.append((cx+radius*RCONST, cy+radius*RCONST))
    ## move left down
    points.append((cx-radius*RCONST, cy+radius*RCONST))
    ## move left up
    points.append((cx-radius*RCONST, cy-radius*RCONST))
    ## move right up
    points.append((cx+radius*RCONST, cy-radius*RCONST))

    return points
    

## checks if circle lies in rectangle 'r'
def circCovered(circle, r):
    ## get the circle's center
    circCenter = (circle.x_pos, circle.y_pos)

    ## create list of points from circle to check
    for p in getCircPoints(circCenter, circle.radius):
        if pointCovered(p, r):
            return True
    return False

## determine width of text
def getTWidth(num):
    if num < 10:
        return 40
    if num < 100:
        return 80
    if num < 1000:
        return 120
    return 160

def get_faces(img):
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=5,
	minSize=(30, 30),
	flags=0
    )
    return faces

# set up the ROI for tracking
#roi = frame[r:r+h, c:c+w]
#hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#roi_hist = doit(frame,r,h,c,w)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
circles = [Circle(cap_width) for _ in xrange(NUM_OF_CIRCLES)]

tracking = False
roi_hist = None

while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ## get width of frame
        height,width,_ = frame.shape

        if tracking:
            cv2.destroyWindow('hsv_start')
            for (x, y, w, h) in get_faces(frame):
                cv2.rectangle(hsv, (x, y), (x+w, y+h), (0, 0, 0), -1)

            ## apply camshift to get the new location
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            #print "dst:",dst
            #print "track_window:",track_window
            #print "term_crit:",term_crit
            rect, track_window = cv2.CamShift(dst, track_window, term_crit)

            pts = cv2.boxPoints(rect)
            pts = np.int0(pts)

            img2 = cv2.polylines(frame, [pts], True, 255, 2)

            # second hsv image
            hsv2 = cv2.polylines(hsv, [pts], True, 255, 2)
            hsv2 = cv2.flip(hsv, flipCode=1)
            cv2.imshow('hsv', hsv2)

            ## bubble fun
            ## ~~~~~~~~~
            #print "--------------"
            #print "rect:",rect
            ## bring 4 rectangle points into correct format
            ## (pts has reversed x-axis?!)
            corPTS = [p.tolist() for p in pts]
            corPTS = [(p[0], p[1]) for p in corPTS]  # [(width-p[0], p[1]) for p in corPTS]
            #print "pts:",pts
            #print "corPTS:",corPTS
            found = False
            for circ in circles:
                if circCovered(circ, corPTS):
                    circ.visible = False
                    circ.popped = True
                    SCORE += 1
                    #print "covered a circle!"
            #print "--------------"
        else:
            ## only draw static rectangle at starting position
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

            hsv2 = cv2.flip(hsv, flipCode=1)
            cv2.imshow('hsv_start', hsv2)


        # for (x, y, w, h) in get_faces(frame):
	    # cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 0), -1)

        ## iterate through circles
        for circ in circles:
            circ.move(img2)
            if not circ.visible:
                ## replace leaving circle with a fresh one
                circles.append(Circle(width))
                circles.remove(circ)
                if not circ.popped:
                    MISSES += 1

        #print "circle count:",len(circles)

        ## flip image
        img2 = cv2.flip(img2, flipCode=1)
        
        ## print score & misses
        #stw = getTWidth(SCORE)
        mtw = getTWidth(MISSES)
        
        cv2.putText(img2, str(SCORE), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
        cv2.putText(img2, str(MISSES), (width-mtw,height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
        
        # Draw it on image
        cv2.imshow('img2',img2)

        key = cv2.waitKey(30)
        if key & 0xFF == ord('s'):
            tracking = True
            roi_hist = startTracking(frame,r,h,c,w)
        elif key & 0xFF == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
