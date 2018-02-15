from tracking.base import Online
from utils import getframes
from tracking.base import Path
import cv2
import dlib
import vision
from face_tracker_DLib import DLibFaceTracker
import numpy as np

TRACKING_THRESHOLD = 0.1

class DlibTrack(Online):
    
    def __init__(self):
        self.tracker = DLibFaceTracker()
        self.started = False
    
    def track(self, pathid, start, stop, basepath, paths):
        if pathid not in paths:
            return Path(None, None, {})

        path = paths[pathid]

        if start not in path.boxes:
            return Path(path.label, path.id, {})
        
        
            
        
        startbox = path.boxes[start]
        initialrect = [startbox.xtl, startbox.ytl, startbox.xbr-startbox.xtl, startbox.ybr-startbox.ytl]
        startbox = [startbox.xtl, startbox.ytl, startbox.xbr, startbox.ybr]
        frames = getframes(basepath, False)
        previmage = frames[start]
        imagesize = previmage.shape
        #print "Frame shape: "
        #print previmage.shape
        #print '=================='
        #if not self.started:
        self.tracker.start_track(frames[start], startbox)#self.visbox_to_dbox(startbox))
        self.started = True
        
        boxes = self.dlib_track(start, stop, frames, initialrect, imagesize)
        #meanshift(start, stop, initialrect, imagesize)
        # boxes need to be in vision.Box() form
        # width is computed from x's, so corresponds to the columns, y to the rows (see documentation in annotation.py
        #[dbox.left(), dbox.top(), dbox.right(), dbox.bottom()]
        return Path(path.label, path.id, boxes)
    
    def dlib_track(self, start, stop, frames, initialrect, imagesize):
        #print "Starting tracking again!"
        boxes = {}
        #points = self.tracker.get_position()
        rect = initialrect#points_to_visbox(points, start)
        inc = 1 if start <= stop else -1
        occluded = 0
        for frame in range(start, stop, inc):
            frame1 = 0
            track_score = self.tracker.update(frames[frame].astype(np.uint8))/100.0
            #print "Score: %f"%(track_score/100.0)
            rect = self.tracker.get_position()
            #print rect
            # If we updated the rect save it and continue to the next frame
            # Otherwise, if the tracking score is too low, call the face occluded and stop
            # Otherwise mark as lost
            
            if not self.insideframe(rect, imagesize):
                
                print "Frame {0} lost in {1} to {2}".format(frame, start, stop)
                
                # make sure bbox is within bounds of the image and all corners
                # are where they should be relative to each other
                xtl = max(0, rect[0])
                ytl = max(0, rect[1])
                xbr = max(min(imagesize[1], rect[0] + initialrect[2]), xtl+1)
                ybr = max(min(imagesize[0], rect[1] + initialrect[3]), ytl+1)
                #print((xtl, ytl, xbr, ybr))
                boxes[frame] = vision.Box(
                    xtl,
                    ytl,
                    xbr,
                    ybr,
                    frame=frame,
                    #lost=True,
                    generated=True,
                    score=track_score
                )
                #break
                """
                print("RECT")
                print(rect)
                print("IMAGESIZE")
                print(imagesize)
                print("INITIALRECT")
                print(initialrect)
                print "May have gone off frame"
                """
            else:
                '''
                if track_score < TRACKING_THRESHOLD:
                    print("Frame {0} occluded in {1} to {2} with score {3}".format(frame, start, stop, track_score))
                    occluded += 1
                    frame1 = frame
                    boxes[frame] = vision.Box(
                        max(0, rect[0]),
                        max(0, rect[1]),
                        min(imagesize[1], rect[0] + initialrect[2]),
                        min(imagesize[0], rect[1] + initialrect[3]),
                        frame=frame,
                        #lost=True,
                        generated=True,
                        occluded=True,
                        score=track_score
                        #attributes=['Occluded']
                    )
                else:
                    frame2 = frame
                    if frame1 == frame2:
                        print("Something isn't right...")
                '''
                
                boxes[frame] = vision.Box(
                    max(0, rect[0]),
                    max(0, rect[1]),
                    min(imagesize[1], rect[0] + initialrect[2]),
                    min(imagesize[0], rect[1] + initialrect[3]),
                    frame=frame,
                    generated=True,
                    score=track_score
                )
    
        #print(occluded)
        
        return boxes
        
    
    def dbox_to_visbox(dbox, frame):
        """
        def __init__(self, int xtl, int ytl, int xbr, int ybr,
                 int frame = 0, int lost = 0, int occluded = 0,
                 image = None, label = None,
                 int generated = 0, double score = 0.0, attributes = None):
        """
        visbox = vision.Box(dbox.left(), dbox.top(), dbox.left(), dbox.bottom(), frame=frame)
        return visbox
    
    def visbox_to_dbox(self, visbox):
        #dbox = dlib.rectangle(long(visbox.xtl), long(visbox.ytl), long(visbox.xbr), long(visbox.ybr))
        return [visbox.xtl, visbox.ytl, visbox.xbr, visbox.ybr]
        
    def points_to_visbox(self, points, frame):
        """
        """
        return vision.Box(points[0], points[1], points[2], points[3], frame=frame)
    
    
    def insideframe(self, rect, imagesize):
        
        return (rect[0] >= 0 and
            rect[1] >= 0 and
            rect[2] <= imagesize[1] and
            rect[3] <= imagesize[0])
    
        
