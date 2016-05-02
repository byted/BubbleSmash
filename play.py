import numpy as np
import cv2
import random

NUM_OF_CIRCLES = 10

cap = cv2.VideoCapture(0)


class Circle(object):
    def __init__(self):
        self.y_pos = -random.randint(0, 100)  # don't let them start all at once
        self.x_pos = random.randint(0, 500)
        self.speed = random.randint(5, 25)
        self.visible = True

    def down(self):
        self.y_pos += self.speed
        if self.y_pos > 500:  # TODO: better way to check if out of frame
            self.visible = False

    def draw(self, video_img):
        cv2.circle(video_img, (self.x_pos, self.y_pos), 10, 255, -1)

    def move(self, video_img):
        self.down()
        if self.visible:
            self.draw(video_img)

cs = [Circle() for _ in xrange(NUM_OF_CIRCLES)]
print cs

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    # ret, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)

    # this is really not pythonic. Need a better way
    for i in xrange(len(cs)):
        cs[i].move(img)
        if not cs[i].visible:
            cs[i] = Circle()

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
