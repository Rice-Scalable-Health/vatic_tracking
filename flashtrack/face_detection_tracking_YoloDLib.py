import os
import sys
import time
import numpy as np
import cv2
import skvideo
skvideo.setFFmpegPath(os.path.dirname(sys.executable))
import skvideo.io
from face_detector_YOLOv2 import YoloFace
from face_tracker_DLib import DLibFaceTracker
from itertools import count

'''
Allows this file to be inported from anywhere, updated the model location to be generalizable
'''
file_path = os.path.realpath(__file__)
model_path = '/YOLO_face_detection/darknet_face_release'
end_ind = file_path.rfind('detection_tracking')#file_path.find(substring, string.find(substring) + 1)
end_ind = file_path[0:end_ind].rfind('detection_tracking')
#print(file_path)
#print("FINAL PATH")
#print(file_path[0:end_ind] + model_path)
#print("#########################################")
yolo_model = YoloFace(file_path[0:end_ind] + model_path)

#print()

#yolo_model = YoloFace('../YOLO_face_detection/darknet_face_release')    

def stdbox2yoloinput(stdbox):
    input_box = dict()
    input_box['left'] = stdbox[0]
    input_box['right'] = stdbox[2]
    input_box['top'] = stdbox[1]
    input_box['bottom'] = stdbox[3]
    return input_box

def draw_rect(img, dboxes,save_file, save_flag, iframe):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel  
    for dbox in dboxes:
        cv2.rectangle(cv_img, (int(dbox[0]), int(dbox[1])), (int(dbox[2]), int(dbox[3])), (0,0,255), 3)      
    
    if save_flag==True:
        cv2.imwrite(save_file, cv_img)     
    
    cv2.putText(cv_img, 'Frame: %d'%(iframe), (20,30),  cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    
    cv_img = cv_img[:,:,[2,1,0]]
    return cv_img

def calc_overlap_ratio(dbox1, dbox2):
    # [left, top, right, bottom]
    min_left = min(dbox1[0], dbox2[0])
    max_left = max(dbox1[0], dbox2[0])
    min_right = min(dbox1[2], dbox2[2])
    max_right = max(dbox1[2], dbox2[2])
    min_top = min(dbox1[1], dbox2[1])
    max_top = max(dbox1[1], dbox2[1])
    min_bottom = min(dbox1[3], dbox2[3])
    max_bottom = max(dbox1[3], dbox2[3])
    overlap_width = max(0, min_right-max_left)
    overlap_height = max(0, min_bottom-max_top)
    overlap_area = overlap_width*overlap_height
    area1 = (dbox1[2]-dbox1[0])*(dbox1[3]-dbox1[1])
    area2 = (dbox2[2]-dbox2[0])*(dbox2[3]-dbox2[1])
    total_area = area1+area2
    ratio = float(overlap_area)/(total_area-overlap_area)
    union_dbox = [min_left, min_top, max_right, max_bottom]    
    
    return ratio, union_dbox
    
def mse_detect(index, interval, frame_queue, img):
    frame_queue.append(img)
  
    if index<interval:
        return 1
    else:
        anchor_img = frame_queue.pop(0)
        mse = np.sum((anchor_img-img)**2)/np.sum((anchor_img)**2, dtype=np.float32)
    #print 'frame%d: mse %f\n' %(index, mse)
    return mse

#def face_detection_tracking_frames(frame, file_dict, flag_dict, param_dict):

def face_detection_tracking_frames_rt(streamer, path_dict, flag_dict, param_dict, start_frame=1, end_frame=-1):
    '''
    '''
    video_output_file =  path_dict['video_output_file'] 
    image_output_folder =  path_dict['image_output_folder']
    log_file = path_dict['log_file']
    output_file = path_dict['output_file']
    #if not os.path.isdir(image_output_folder):
    #    os.makedirs(image_output_folder)
    if not os.path.isdir(os.path.dirname(output_file)):
        print(os.path.dirname(output_file))
        os.makedirs(os.path.dirname(output_file))
           
    # prepare flags   
    write_video_flag = flag_dict['write_video_flag'] 
    write_image_flag = True#flag_dict['write_image_flag'] 
    write_log_flag = flag_dict['write_log_flag'] 
    write_output_flag = flag_dict['write_output_flag']
    if write_video_flag==True:
        if os.path.exists(video_output_file):
            os.remove(video_output_file)
        video_writer = skvideo.io.FFmpegWriter(video_output_file, inputdict={'-r':'30'})          
    if write_log_flag==True:
        if os.path.exists(log_file):
            os.remove(log_file)           
              
    # prepare trade-off parameters
    mse_thresh = param_dict['mse_thresh'] 
    track_thresh = param_dict['track_thresh'] 
    detect_thresh = param_dict['detect_thresh'] 
    track_add_thresh = param_dict['track_add_thresh']
    w = param_dict['width']
    h = param_dict['height']
    
    # remaining initialization
    interval = 1
    total_frame_num = 0
    detect_frame_num = 0
    track_update_face_num = 1e-10
    track_start_face_num = 0
    track_detect_num = 1e-10
    total_time = 0.0
    total_track_update_time = 0.0
    total_track_start_time = 0.0
    total_detect_time = 0.0
    total_track_detect_time = 0.0
    start_frame = 1
    end_frame = -1    
    trackers = []
    frame_queue = []        
    iframe = start_frame-1    

    with open(path_dict['log_file'], 'a+') as ft, open(path_dict['output_file'], 'a+') as fa:

        for k in count():

            img_bytes = streamer.stdout.read(w*h*3)
            if len(img_bytes) < w*h*3:
                break
            img = np.frombuffer(img_bytes, dtype=np.uint8, count=w*h*3).reshape(h, w, 3)
            img = np.array(img)
            
            iframe += 1   
            if iframe==end_frame:
                break
              
            save_image_file = os.path.join('frames', 'frame_%09d.png'%(iframe))
              
            total_frame_num += 1
            t1 = time.time()
            track_scores = []
            dboxes = []
              
            if len(trackers)>0:
                track_update_face_num += len(trackers)
                for i in range(len(trackers)):
                    track_score = trackers[i].update(img)
                    track_scores.append(track_score/100.0)
                    tbox = trackers[i].get_position()
                    dboxes.append(tbox)

            #conf_scores = np.copy(track_scores)
            total_track_update_time += time.time()-t1
              
            # redetect face
            mse = mse_detect(iframe-start_frame, interval, frame_queue, img) 
              
            detect_flag = False
            if len(trackers)==0:
                if mse>mse_thresh:
                    detect_flag=True
            else:
                if min(track_scores)<track_thresh or mse>mse_thresh:
                    detect_flag=True
              
            if detect_flag==True:
                detect_frame_num += 1           
                tdetect = time.time()         
                yolo_boxes, _= yolo_model.yolo_detect_face(img, thresh=detect_thresh)
                total_detect_time+=time.time()-tdetect
                  
                dboxes = []
                conf_scores = []
                for yolo_box in yolo_boxes:
                    dbox = [yolo_box['left'], yolo_box['top'], yolo_box['right'], yolo_box['bottom']]    
                    dboxes.append(dbox)
                    conf_scores.append(yolo_box['prob'])
               
                if len(trackers)>0:
                    sort_idx = np.argsort(track_scores)
                    sort_idx = sort_idx[::-1]
                    trackers = [trackers[i] for i in sort_idx]
                    track_scores = [track_scores[i] for i in sort_idx]
                
                    for i in range(len(trackers)):
                        if track_scores[i]<track_thresh:
                            break
                        tbox = trackers[i].get_position()                            
                          
                        overlap_flag = False
                        for di, dbox in enumerate(dboxes):
                            overlap_ratio,union_dbox = calc_overlap_ratio(dbox, tbox)
                            if overlap_ratio>0:
                                overlap_flag = True
                                if conf_scores[di]<detect_thresh+0.1:
                                    dboxes[di] = union_dbox
                                break
                          
                        if overlap_flag==False:
                            ttrack_detect = time.time()
                            track_detect_num += 1
                            track_detect_scores, corrected_box_coords, _ = yolo_model.yolo_check_boxes(img.shape[0], img.shape[1], [stdbox2yoloinput(tbox)]) 
                            #print 'track_detect_score '+str(track_detect_scores[0])
                            if track_detect_scores[0]>track_add_thresh:
                                conf_scores.append(track_detect_scores[0])
                                if track_detect_scores[0]>(detect_thresh+track_add_thresh)/2:
                                    corrected_tbox = corrected_box_coords[0].tolist()
                                    dboxes.append(corrected_tbox)
                                else:
                                    dboxes.append(tbox)
                                  
                            total_track_detect_time += time.time()-ttrack_detect
                              
                ttrack_start = time.time()
                trackers = [DLibFaceTracker() for i in range(len(dboxes))]                        
                for i, dbox in enumerate(dboxes):
                    track_start_face_num += 1
                    trackers[i].start_track(img, dbox)
                      
                total_track_start_time += time.time()-ttrack_start
            #total_detect_time += time.time()-t2
            total_time += time.time()-t1
                  
            if write_log_flag==True:

                ft.write('frame: %d detect_ratio: %f avg_time: %fs\n' %(iframe, float(detect_frame_num)/total_frame_num, total_time/total_frame_num))
                ft.write('avg_detection_time: %fs\n' % (total_detect_time/detect_frame_num))
                ft.write('avg_tracking_update_time: %fs\n' % (total_track_update_time/track_update_face_num))
                if not track_start_face_num == 0 :
                    ft.write('avg_tracking_start_time: %fs\n' %(total_track_start_time/track_start_face_num))
                ft.write('avg_track-detect_time: %fs\n' %( total_track_detect_time/track_detect_num))
              
            for i, dbox in enumerate(dboxes):
                fa.write('%d %d %d %d %d\n'%(dbox[0], dbox[1], dbox[2], dbox[3], iframe))
                
            if write_video_flag==True:                 
                bb_img = draw_rect(img, dboxes, save_image_file, write_image_flag, iframe)
                print("WRITING FRAME")
                video_writer.writeFrame(bb_img)

        if write_video_flag==True:
            video_writer.close()  


def face_detection_tracking(path_dict,  flag_dict, param_dict, start_frame=1, end_frame=-1):

    # prepare paths
    video_input_file = path_dict['video_input_file'] 
    video_output_file =  path_dict['video_output_file'] 
    image_output_folder =  path_dict['image_output_folder']
    log_file = path_dict['log_file']
    output_file = path_dict['output_file']
    if not os.path.isdir(image_output_folder):
        os.makedirs(image_output_folder)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    videogen = skvideo.io.vreader(video_input_file)
           
    # prepare flags   
    write_video_flag = flag_dict['write_video_flag'] 
    write_image_flag = flag_dict['write_image_flag'] 
    write_log_flag = flag_dict['write_log_flag'] 
    write_output_flag = flag_dict['write_output_flag']
    if write_video_flag==True:
        if os.path.exists(video_output_file):
            os.remove(video_output_file)
        video_writer = skvideo.io.FFmpegWriter(video_output_file)          
    if write_log_flag==True:
        if os.path.exists(log_file):
            os.remove(log_file)           
              
    # prepare trade-off parameters
    mse_thresh = param_dict['mse_thresh'] 
    track_thresh = param_dict['track_thresh'] 
    detect_thresh = param_dict['detect_thresh'] 
    track_add_thresh = param_dict['track_add_thresh']
    
    # remaining initialization
    interval = 1
    total_frame_num = 0
    detect_frame_num = 0
    track_update_face_num = 1e-10
    track_start_face_num = 0
    track_detect_num = 1e-10
    total_time = 0.0
    total_track_update_time = 0.0
    total_track_start_time = 0.0
    total_detect_time = 0.0
    total_track_detect_time = 0.0
        
    trackers = []
    frame_queue = []        
    iframe = start_frame-1    

    ###############################
    ##main code starts here
    for img in videogen:
    
        iframe += 1   
        if iframe==end_frame:
            break
          
        save_image_file = os.path.join(image_output_folder, 'frame_%09d.png'%(iframe))
          
        total_frame_num += 1
        t1 = time.time()
        track_scores = []
        dboxes = []
          
        if len(trackers)>0:
            track_update_face_num += len(trackers)
            for i in range(len(trackers)):
                track_score = trackers[i].update(img)
                track_scores.append(track_score/100.0)
                tbox = trackers[i].get_position()
                dboxes.append(tbox)
    
        #conf_scores = np.copy(track_scores)
        total_track_update_time += time.time()-t1
          
        # redetect face
        mse = mse_detect(iframe-start_frame, interval, frame_queue, img) 
          
        detect_flag = False
        if len(trackers)==0:
            if mse>mse_thresh:
                detect_flag=True
        else:
            if min(track_scores)<track_thresh or mse>mse_thresh:
                detect_flag=True
          
        if detect_flag==True:
            detect_frame_num += 1           
            tdetect = time.time()         
            yolo_boxes, _= yolo_model.yolo_detect_face(img, thresh=detect_thresh)
            total_detect_time+=time.time()-tdetect
              
            dboxes = []
            conf_scores = []
            for yolo_box in yolo_boxes:
                dbox = [yolo_box['left'], yolo_box['top'], yolo_box['right'], yolo_box['bottom']]    
                dboxes.append(dbox)
                conf_scores.append(yolo_box['prob'])
           
            if len(trackers)>0:
                sort_idx = np.argsort(track_scores)
                sort_idx = sort_idx[::-1]
                trackers = [trackers[i] for i in sort_idx]
                track_scores = [track_scores[i] for i in sort_idx]
            
                for i in range(len(trackers)):
                    if track_scores[i]<track_thresh:
                        break
                    tbox = trackers[i].get_position()                            
                      
                    overlap_flag = False
                    for di, dbox in enumerate(dboxes):
                        overlap_ratio,union_dbox = calc_overlap_ratio(dbox, tbox)
                        if overlap_ratio>0:
                            overlap_flag = True
                            if conf_scores[di]<detect_thresh+0.1:
                                dboxes[di] = union_dbox
                            break
                      
                    if overlap_flag==False:
                        ttrack_detect = time.time()
                        track_detect_num += 1
                        track_detect_scores, corrected_box_coords, _ = yolo_model.yolo_check_boxes(img.shape[0], img.shape[1], [stdbox2yoloinput(tbox)]) 
                        #print 'track_detect_score '+str(track_detect_scores[0])
                        if track_detect_scores[0]>track_add_thresh:
                            conf_scores.append(track_detect_scores[0])
                            if track_detect_scores[0]>(detect_thresh+track_add_thresh)/2:
                                corrected_tbox = corrected_box_coords[0].tolist()
                                dboxes.append(corrected_tbox)
                            else:
                                dboxes.append(tbox)
                              
                        total_track_detect_time += time.time()-ttrack_detect
                          
            ttrack_start = time.time()
            trackers = [DLibFaceTracker() for i in range(len(dboxes))]                        
            for i, dbox in enumerate(dboxes):
                track_start_face_num += 1
                trackers[i].start_track(img, dbox)
                  
            total_track_start_time += time.time()-ttrack_start
        #total_detect_time += time.time()-t2
        total_time += time.time()-t1
          
        print 'frame: %d detect_ratio: %f avg_time: %f' %(iframe, float(detect_frame_num)/total_frame_num, total_time/total_frame_num)
        print 'avg_detection_time: %fs' % (total_detect_time/detect_frame_num)
        print 'avg_tracking_update_time: %fs' % (total_track_update_time/track_update_face_num)
        if not track_start_face_num == 0:
            print 'avg_tracking_start_time: %fs' %(total_track_start_time/track_start_face_num)
              
        if write_log_flag==True and not track_start_face_num == 0:
            ft = open(log_file, 'a+')
            ft.write('frame: %d detect_ratio: %f avg_time: %fs\n' %(iframe, float(detect_frame_num)/total_frame_num, total_time/total_frame_num))
            ft.write('avg_detection_time: %fs\n' % (total_detect_time/detect_frame_num))
            ft.write('avg_tracking_update_time: %fs\n' % (total_track_update_time/track_update_face_num))
            ft.write('avg_tracking_start_time: %fs\n' %(total_track_start_time/track_start_face_num))
            ft.write('avg_track-detect_time: %fs\n' %( total_track_detect_time/track_detect_num))
            ft.close()
          
        if write_output_flag==True:
            fa = open(output_file, 'a+')
            for i, dbox in enumerate(dboxes):
                fa.write('%d %d %d %d %d\n'%(dbox[0], dbox[1], dbox[2], dbox[3], iframe))
            fa.close()
            
        if write_video_flag==True:                 
            bb_img = draw_rect(img, dboxes, save_image_file, write_image_flag, iframe)
            video_writer.writeFrame(bb_img)

    if write_video_flag==True:
        video_writer.close()   
