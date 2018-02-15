import dlib
from face_tracker import FaceTracker

class DLibFaceTracker(FaceTracker):
    
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
        
    def update(self, img):
        return self.tracker.update(img)

    def start_track(self, img, box):
        dbox = dlib.rectangle(long(box[0]), long(box[1]), long(box[2]), long(box[3]))    
        self.tracker.start_track(img, dbox)
        
    def get_position(self):
        dbox = self.tracker.get_position()
        return [dbox.left(), dbox.top(), dbox.right(), dbox.bottom()]