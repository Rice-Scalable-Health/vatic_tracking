from abc import abstractmethod

class FaceTracker(object):
    @abstractmethod
    def update(self, img):
        '''
        update the tracking box based on current frame
        '''
    @abstractmethod
    def start_track(self, img, box):
        """
        start tracking given an image and a bounding box in it
        """
    @abstractmethod
    def get_position(self, img):
        """
        get the position (left, top, right, bottom) of tracking box
        """
    
        

