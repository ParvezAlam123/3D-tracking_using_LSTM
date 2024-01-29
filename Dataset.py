import numpy as np 
import struct 
import os 
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset, DataLoader 







def load_kitti_calib(calib_file):
   
   with open(calib_file) as f_calib:
      lines = f_calib.readlines()
      
   P0 = np.array(lines[0].strip('\n').split()[1:], dtype=np.float32)
   P1 = np.array(lines[1].strip('\n').split()[1:], dtype=np.float32)
   P2 = np.array(lines[2].strip('\n').split()[1:], dtype=np.float32)
   P3 = np.array(lines[3].strip('\n').split()[1:], dtype=np.float32)
   R0_rect = np.array(lines[4].strip('\n').split()[1:], dtype=np.float32)
   Tr_velo_to_cam = np.array(lines[5].strip('\n').split()[1:], dtype=np.float32)
   Tr_imu_to_velo = np.array(lines[6].strip('\n').split()[1:], dtype=np.float32) 
   
   return {'P0': P0, 'P1':P1, 'P2':P2, 'P3':P3, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3,4), 'Tr_imu_to_velo': Tr_imu_to_velo}
   
   
 
 
 
def camera_coordinate_to_point_cloud(box3d, Tr):

   def project_cam2velo(cam, Tr):
      T = np.zeros([4,4], dtype=np.float32)
      T[:3, :] = Tr 
      T[3, 3] = 1 
     
      T_inv = np.linalg.inv(T) 
      lidar_loc_ = np.dot(T_inv, cam) 
      lidar_loc = lidar_loc_[:3]
      
      return lidar_loc.reshape(1,3) 
      
   def ry_to_rz(ry):
      angle = -ry - np.pi / 2
      
      if angle >= np.pi:
         angle -= np.pi 
      if angle < -np.pi:
         angle = 2 * np.pi + angle 
      return angle 
      
      
      
   
   h,w,l,tx,ty,tz,ry = [float(i) for i in box3d]
   cam = np.ones([4,1])
   cam[0] = tx 
   cam[1] = ty
   cam[2] = tz 
   t_lidar = project_cam2velo(cam, Tr) 
  
   
   Box = np.array([[-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                   [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                   [0, 0, 0, 0, h, h,  h, h]])
                   
   rz = ry_to_rz(ry) 
   
   rotMat = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                       [np.sin(rz), np.cos(rz), 0.0],
                       [0.0,          0.0,       1.0]])
   
                      
   velo_box = np.dot(rotMat, Box) 
     
   cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T 
   
   box3d_corner = cornerPosInVelo.transpose() 
   
   return t_lidar, box3d_corner, h, w, l , rz 
   
   



class Data(Dataset):
   def __init__(self, velo_dir, label_dir, calib_dir, train=True):
      self.label_dir = label_dir
      self.calib_dir = calib_dir
      self.velo_dir = velo_dir 
      self.train=train

      self.calibs = sorted(os.listdir(self.label_dir))
      self.scenes = sorted(os.listdir(self.velo_dir))
      self.labels = sorted(os.listdir(self.label_dir)) 

      

      self.files = [] 


      if self.train:
         train_scenes = []
         train_calibs = []
         train_labels = [] 
         
         train_scenes.append(self.scenes[0])
         train_scenes.append(self.scenes[1])
         train_scenes.append(self.scenes[2])
         train_scenes.append(self.scenes[3])
         train_scenes.append(self.scenes[4])
         train_scenes.append(self.scenes[5]) 
         train_scenes.append(self.scenes[6])
         train_scenes.append(self.scenes[7])
         train_scenes.append(self.scenes[8])
         train_scenes.append(self.scenes[9])
         train_scenes.append(self.scenes[10])
         train_scenes.append(self.scenes[11])
         train_scenes.append(self.scenes[12])
         train_scenes.append(self.scenes[13])
         train_scenes.append(self.scenes[14])
         train_scenes.append(self.scenes[15])
         train_scenes.append(self.scenes[16])
         train_scenes.append(self.scenes[17])
         train_scenes.append(self.scenes[20])

         train_calibs.append(self.calibs[0])
         train_calibs.append(self.calibs[1])
         train_calibs.append(self.calibs[2])
         train_calibs.append(self.calibs[3])
         train_calibs.append(self.calibs[4])
         train_calibs.append(self.calibs[5])
         train_calibs.append(self.calibs[6])
         train_calibs.append(self.calibs[7])
         train_calibs.append(self.calibs[8])
         train_calibs.append(self.calibs[9])
         train_calibs.append(self.calibs[10])
         train_calibs.append(self.calibs[11])
         train_calibs.append(self.calibs[12])
         train_calibs.append(self.calibs[13])
         train_calibs.append(self.calibs[14])
         train_calibs.append(self.calibs[15])
         train_calibs.append(self.calibs[16])
         train_calibs.append(self.calibs[17])
         train_calibs.append(self.calibs[20])

         train_labels.append(self.labels[0])
         train_labels.append(self.labels[1])
         train_labels.append(self.labels[2])
         train_labels.append(self.labels[3])
         train_labels.append(self.labels[4])
         train_labels.append(self.labels[5])
         train_labels.append(self.labels[6])
         train_labels.append(self.labels[7])
         train_labels.append(self.labels[8])
         train_labels.append(self.labels[9])
         train_labels.append(self.labels[10])
         train_labels.append(self.labels[11])
         train_labels.append(self.labels[12])
         train_labels.append(self.labels[13])
         train_labels.append(self.labels[14])
         train_labels.append(self.labels[15])
         train_labels.append(self.labels[16])
         train_labels.append(self.labels[17])
         train_labels.append(self.labels[20])


         for i in range(len(train_scenes)):
             pcd_file_path = os.path.join(self.velo_dir, train_scenes[i])
             calib_file = os.path.join(self.calib_dir, train_calibs[i]) 
             label_file = os.path.join(self.label_dir, train_labels[i])  
             calibration = load_kitti_calib(calib_file)
    
    
    
    
             # get the total number of frames in particular scene 
             num_frames = len(os.listdir(pcd_file_path)) 
    
    
             bb_list = []            # store bounding boxex of complete scene 
    
             with open(label_file) as f_label:
                lines = f_label.readlines()
       
                for line in lines:
                   line = line.strip('\n').split() 
                   if line[2] != 'DontCare':
                      frame_index = line[0]             # frame number
                      cat_id = line[1]                  # cat id 
                      category = line[2]                # class of the BB 
                      center, box3d_corner, h, w, l, rz = camera_coordinate_to_point_cloud(line[10:17], calibration['Tr_velo_to_cam'])
                      center = center[0] 
                      bb_list.append([frame_index, cat_id, center, h, w, l, rz])
          
       
    
             pcd_frames = sorted(os.listdir(pcd_file_path)) 
          
             n = 0 
             while n < len(pcd_frames):
                sample = {} 
                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n : 
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]]) 

                sample["bb_first_frame"] = bboxes  
             
                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+1 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_second_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+2 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_third_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+3 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_fourth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+4 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_fifth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+5 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_sixth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+6 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_seventh_frame"] = bboxes
             
                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+7 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_eighth_frame"] = bboxes

             
                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+8 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_ninth_frame"] = bboxes 


                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+9 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_tenth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+10 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_eleventh_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+11 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_twelveth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+12 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_thirteenth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+13 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_fourteenth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+14 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_fifteenth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+15 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_sixteenth_frame"] = bboxes 

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+16 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_seventeenth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+17 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_eighteenth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+18 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_ninteenth_frame"] = bboxes  

                bboxes = [] 
                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+19 :
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6]])
             
                sample["bb_twentyth_frame"] = bboxes  


                n = n + 20

                self.files.append(sample)
      else:
         valid_scenes = []
         valid_calibs = [] 
         valid_labels = [] 


        
         #valid_scenes.append(self.scenes[1])
         #valid_scenes.append(self.scenes[6])
         #valid_scenes.append(self.scenes[8])
         #valid_scenes.append(self.scenes[10])
         #valid_scenes.append(self.scenes[12])
         #valid_scenes.append(self.scenes[13])
         #valid_scenes.append(self.scenes[14])
         #valid_scenes.append(self.scenes[15])
         #valid_scenes.append(self.scenes[16])
         valid_scenes.append(self.scenes[18])
         valid_scenes.append(self.scenes[19]) 

         #valid_calibs.append(self.calibs[1])
         #valid_calibs.append(self.calibs[6])
         #valid_calibs.append(self.calibs[8])
         #valid_calibs.append(self.calibs[10])
         #valid_calibs.append(self.calibs[12])
         #valid_calibs.append(self.calibs[13])
         #valid_calibs.append(self.calibs[14])
         #valid_calibs.append(self.calibs[15])
         #valid_calibs.append(self.calibs[16])
         valid_calibs.append(self.calibs[18])
         valid_calibs.append(self.calibs[19])

         #valid_labels.append(self.labels[1])
         #valid_labels.append(self.labels[6])
         #valid_labels.append(self.labels[8])
         #valid_labels.append(self.labels[10])
         #valid_labels.append(self.labels[12])
         #valid_labels.append(self.labels[13])
         #valid_labels.append(self.labels[14])
         #valid_labels.append(self.labels[15])
         #valid_labels.append(self.labels[16])
         valid_labels.append(self.labels[18])
         valid_labels.append(self.labels[19])






         for i in range(len(valid_scenes)):
             pcd_file_path = os.path.join(self.velo_dir, valid_scenes[i])
             calib_file = os.path.join(self.calib_dir, valid_calibs[i]) 
             label_file = os.path.join(self.label_dir, valid_labels[i])  
             calibration = load_kitti_calib(calib_file)
    
    
    
    
             # get the total number of frames in particular scene 
             num_frames = len(os.listdir(pcd_file_path)) 
    
    
             bb_list = []            # store bounding boxex of complete scene 
    
             with open(label_file) as f_label:
                lines = f_label.readlines()
       
                for line in lines:
                   line = line.strip('\n').split() 
                   if line[2] != 'DontCare':
                      frame_index = line[0]             # frame number
                      cat_id = line[1]                  # cat id 
                      category = line[2]                # class of the BB 
                      center, box3d_corner, h, w, l, rz = camera_coordinate_to_point_cloud(line[10:17], calibration['Tr_velo_to_cam'])
                      center = center[0] 
                      bb_list.append([frame_index, cat_id, center, h, w, l, rz, box3d_corner,category])
          
       
    
             pcd_frames = sorted(os.listdir(pcd_file_path)) 
          
                
             n = 0 
             while n < len(pcd_frames) // 2:
                bboxes = []
                sample = {} 

                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n : 
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6], bb_list[k][7], bb_list[k][8]]) 
                
                sample["bb_first_frame"] = bboxes  
                sample["first_pcd_path"] = os.path.join(pcd_file_path, pcd_frames[n]) 

                bboxes = []

                for k in range(len(bb_list)):
                   if int(bb_list[k][0]) == n+1 : 
                      bboxes.append([bb_list[k][1], bb_list[k][2], bb_list[k][3], bb_list[k][4], bb_list[k][5], bb_list[k][6], bb_list[k][7], bb_list[k][8]]) 
                
                sample["bb_second_frame"] = bboxes  
                sample["second_pcd_path"] = os.path.join(pcd_file_path, pcd_frames[n]) 

                n = n + 2 


                self.files.append(sample)



                




   def path_to_points(self, pcd_path):
       size_float = 4 
       list_pcd = [] 
       with open(pcd_path, "rb") as f:
          byte = f.read(size_float * 4) 
          while byte:
             x, y, z, intensity = struct.unpack("ffff", byte) 
             list_pcd.append([x,y,z]) 
             byte = f.read(size_float * 4) 
       points = np.asarray(list_pcd) 
       return points 
   





   
   def __len__(self):
      return len(self.files)
   
   
   def __getitem__(self, index):
      first_frame = self.files[index]["bb_first_frame"] 
      second_frame = self.files[index]["bb_second_frame"]
   
      first_pcd_path = self.files[index]["first_pcd_path"]
      second_pcd_path = self.files[index]["second_pcd_path"]
      points1 = self.path_to_points(first_pcd_path)
      points2 = self.path_to_points(second_pcd_path)
      



      #third_frame = self.files[index]["bb_third_frame"]
      #fourth_frame = self.files[index]["bb_fourth_frame"]
      #fifth_frame = self.files[index]["bb_fifth_frame"]
      #sixth_frame = self.files[index]["bb_sixth_frame"]
      #seventh_frame = self.files[index]["bb_seventh_frame"]
      #eighth_frame = self.files[index]["bb_eighth_frame"]
      #ninth_frame = self.files[index]["bb_ninth_frame"]
      #tenth_frame = self.files[index]["bb_tenth_frame"]
      #eleventh_frame = self.files[index]["bb_eleventh_frame"]
      #twelveth_frame = self.files[index]["bb_twelveth_frame"]
      #thirteenth_frame = self.files[index]["bb_thirteenth_frame"]
      #fourteenth_frame = self.files[index]["bb_fourteenth_frame"]
      #fifteenth_frame = self.files[index]["bb_fifteenth_frame"]
      #sixteenth_frame = self.files[index]["bb_sixteenth_frame"]
      #seventeenth_frame = self.files[index]["bb_seventeenth_frame"]
      #eighteenth_frame = self.files[index]["bb_eighteenth_frame"]
      #ninteenth_frame = self.files[index]["bb_ninteenth_frame"]
      #twentyth_frame = self.files[index]["bb_twentyth_frame"]
      
      #if self.train:
      #return {"first_frame":first_frame, "second_frame":second_frame, "third_frame":third_frame, "fourth_frame":fourth_frame, "fifth_frame":fifth_frame,
      #        "sixth_frame":sixth_frame, "seventh_frame":seventh_frame, "eighth_frame":eighth_frame, "ninth_frame":ninth_frame, "tenth_frame":tenth_frame,
      #       "eleventh_frame":eleventh_frame, "twelveth_frame":twelveth_frame, "thirteenth_frame":thirteenth_frame, "fourteenth_frame":fourteenth_frame,
      #        "fifteenth_frame":fifteenth_frame, "sixteenth_frame":sixteenth_frame, "seventeenth_frame":seventeenth_frame,
      #        "eighteenth_frame":eighteenth_frame, "ninteenth_frame":ninteenth_frame, "twentyth_frame":twentyth_frame}
      #else:
      return {"first_frame":first_frame,  "points1":points1, "second_frame":second_frame, "points2":points2} 
   
   
   

   

   


   

   

   

   

   

   

      
      

   
   


   




   

   


           
             
             
             

     


     
