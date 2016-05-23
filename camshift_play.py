import numpy as np
import cv2
import random
import sys

LEVEL = 1
#LSTAGES = [0, 25, 75, 175, 300, 500]
LSTAGES = [0, 10, 20, 30, 40, 50]

NUM_OF_CIRCLES = 5
SCORE = 0
MISSES = 0
HEARTS = 4

## CIRCLES
class Circle(object):
    def __init__(self, frame_width):
        self.y_pos = -random.randint(0, 100)  # don't let them start all at once
        self.x_pos = random.randint(0, frame_width)
        self.visible = True
        self.popped = False

        ## adjust color to level
        if LEVEL < 4:
            self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        else:
            self.color = (75,75,75)

        ## adjust speed to level
        if LEVEL == 1:
            self.speed = random.randint(5, 10)
        elif LEVEL < 4:
            self.speed = random.randint(10, 15)
        else:
            self.speed = random.randint(12, 25)

        ## adjust radius to level
        if LEVEL == 1:
            self.radius = 30
        elif LEVEL < 3:
            self.radius = 23
        else:
            self.radius = 15

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



WIN = False
LOSS = False

##cascPath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

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

def get_faces(img, cascPath):
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
            for (x, y, w, h) in get_faces(frame, cascPath):
                cv2.rectangle(hsv, (x, y), (x + w, y + int(h * 1.3)), (0, 0, 0), -1)

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


            ## iterate through circles
            for circ in circles:
                circ.move(img2)
                if not circ.visible:
                    ## replace leaving circle with a fresh one
                    circles.append(Circle(width))
                    circles.remove(circ)
                    if not circ.popped:
                        MISSES += 1
                        if MISSES == HEARTS:
                            LOSS = True

            #print "circle count:",len(circles)


            for circ in circles:
                if circCovered(circ, corPTS):
                    #print "covered a circle!"
                    ## Event: Scored a bubble
                    # remove bubble
                    circ.visible = False
                    circ.popped = True
                    # update score
                    SCORE += 1
                    # upgrade level
                    if SCORE == LSTAGES[LEVEL]:
                        LEVEL += 1
                        if LEVEL == len(LSTAGES):
                            print "you win!"
                            WIN = True

            #print "--------------"
        else:
            ## only draw static rectangle at starting position
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)

            hsv2 = cv2.flip(hsv, flipCode=1)
            cv2.imshow('hsv_start', hsv2)


        # for (x, y, w, h) in get_faces(frame):
	    # cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 0), -1)


        ## flip image (IF LEVEL IS LOW)
        if LEVEL < 5:
            img2 = cv2.flip(img2, flipCode=1)

        ## CHECK END CONDITIONS
        #if SCORE > 1:
        #    WIN = True
        if WIN:
            cv2.putText(img2, "You WIN! \o/", (width/5,height/2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255))
        elif LOSS:
            cv2.putText(img2, "You LOSE! :(", (width/4,height/2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))


        ## CHANGE COLOR
        if LEVEL >= 4:
            ## apply gray filter
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


        ## DRAWING FROM HERE

        ## PREPARE HEARTS
        ## PREPARE HEART IMAGE

        ## size of a heart
        heartSize = 50
        ## show hearts/lives
        ## load image
        heartIMG = cv2.imread('images/heart.png', -1)
        ## create mask (why?)
        heart_mask = heartIMG[:,:,3]
        ## create inverted mask (why?)
        heart_mask_inv = cv2.bitwise_not(heart_mask)
        ## convert img to BGR
        heartIMG = heartIMG[:,:,0:3]
        heart = cv2.resize(heartIMG, (heartSize, heartSize), interpolation=cv2.INTER_AREA)
        heartM =cv2.resize(heart_mask, (heartSize, heartSize), interpolation=cv2.INTER_AREA)
        heartM_inv = cv2.resize(heart_mask_inv, (heartSize, heartSize), interpolation=cv2.INTER_AREA)

        ## distance to image boundary
        offset = 10

        ## DRAW HEARTS
        for i in range(HEARTS-MISSES): ## how many left?
            ## DRAW HEART
            width_offset = (i+1)*offset + i*heartSize
            ## take ROI from background
            roiH = img2[height-(heartSize+offset):height-offset, width-(heartSize+width_offset):width-width_offset]
            roi_bgH = cv2.bitwise_and(roiH, roiH, mask = heartM_inv)
            roi_fgH = cv2.bitwise_and(heart, heart, mask = heartM)
            dstH = cv2.add(roi_bgH, roi_fgH)
            img2[height-(heartSize+offset):height-offset, width-(heartSize+width_offset):width-width_offset] = dstH



        ## TEXT: SCORES/LEVELS
        ## print score & misses
        #stw = getTWidth(SCORE)
        mtw = getTWidth(MISSES)
        levelMSG = "Level "+str(LEVEL)

        cv2.putText(img2, str(SCORE), (20,height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
        cv2.putText(img2, levelMSG, ((width/2)-170, height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,185,255))
        #cv2.putText(img2, str(MISSES), (width-mtw,height-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))

        # Draw it on image
        cv2.imshow('img2',img2)

        if WIN or LOSS:
            cv2.waitKey(15000)
            break

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
