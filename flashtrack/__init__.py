from flashtracking import DlibTrack
#from meanshift import MeanShift
#from opticalflow import OpticalFlow
#from bidirectionaloptflow import BidirectionalOptFlow
#from backgroundsubtract import BackgroundSubtract
#from templatematching import TemplateMatch, BidirectionalTemplateMatch

online = {"Flash Tracking": DlibTrack}
#online = {
#    "Optical Flow": OpticalFlow,
#    "Background Subtraction":BackgroundSubtract,
#    "Template": TemplateMatch,
#}
bidirectional = {}
#bidirectional = {"Optical Flow": BidirectionalOptFlow, "Template": BidirectionalTemplateMatch}
multiobject = {}
