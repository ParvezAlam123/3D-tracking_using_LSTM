import numpy as np 
import struct 
import os 
import torch.nn as nn 
import torch 
from torch.utils.data import Dataset, DataLoader 
import json 
import struct 
from pyquaternion import Quaternion  
import copy 
import time 
import open3d as o3d 




class Nuscene:
    def __init__(self, root):
        self.root = root 
        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 
                          'instance', 'log', 'map', 'sample', 'sample_annotation', 'sample_data', 'scene', 'sensor', 'visibility']
        self.attribute = self.read_table('attribute.json')
        self.calibrated_sensor = self.read_table('calibrated_sensor.json')
        self.category = self.read_table('category.json')
        self.ego_pose = self.read_table('ego_pose.json')
        self.instance = self.read_table('instance.json')
        self.log = self.read_table('log.json')
        self.map = self.read_table('map.json')
        self.sample = self.read_table('sample.json')
        self.sample_annotation = self.read_table('sample_annotation.json')
        self.sample_data = self.read_table('sample_data.json')
        self.scene = self.read_table('scene.json')
        self.sensor = self.read_table('sensor.json')
        self.visibility = self.read_table('visibility.json')

        self.token2ind = self.token2ind()
        self.sample_decorate() 





    def read_table(self, table_name):
        path = os.path.join(self.root, table_name)
        f = open(path, 'r')
        file = f.read() 
        table = json.loads(file)
        return table 
    

    def token2ind(self):
        token2ind = {}
        for i in range(len(self.table_names)):
            token2ind[self.table_names[i]] = {}

        for i in range(len(self.attribute)):
            token2ind['attribute'][self.attribute[i]['token']] = i 

        for i in range(len(self.calibrated_sensor)):
            token2ind['calibrated_sensor'][self.calibrated_sensor[i]['token']] = i 

        for i in range(len(self.category)):
            token2ind['category'][self.category[i]['token']] = i 

        for i in range(len(self.ego_pose)):
            token2ind['ego_pose'][self.ego_pose[i]['token']] = i 

        for i in range(len(self.instance)):
            token2ind['instance'][self.instance[i]['token']] = i 

        for i in range(len(self.log)):
            token2ind['log'][self.log[i]['token']] = i 

        for i in range(len(self.map)):
            token2ind['map'][self.map[i]['token']] = i 

        for i in range(len(self.sample)):
            token2ind['sample'][self.sample[i]['token']] = i 

        for i in range(len(self.sample_annotation)):
            token2ind['sample_annotation'][self.sample_annotation[i]['token']] = i 

        for i in range(len(self.sample_data)):
            token2ind['sample_data'][self.sample_data[i]['token']] = i 

        for i in range(len(self.scene)):
            token2ind['scene'][self.scene[i]['token']] = i 

        for i in range(len(self.sensor)):
            token2ind['sensor'][self.sensor[i]['token']] = i 

        for i in range(len(self.visibility)):
            token2ind['visibility'][self.visibility[i]['token']] = i 

        
        return token2ind 
    
    
    def get(self, table_name, token):
        
        if table_name == 'attribute':
            return self.attribute[self.token2ind['attribute'][token]]
        
        if table_name == 'calibrated_sensor':
            return self.calibrated_sensor[self.token2ind['calibrated_sensor'][token]]
        
        if table_name == 'category':
            return self.category[self.token2ind['category'][token]]
        
        if table_name == 'ego_pose':
            return self.ego_pose[self.token2ind['ego_pose'][token]]
        
        if table_name == 'instance':
            return self.instance[self.token2ind['instance'][token]]
        
        if table_name == 'log':
            return self.log[self.token2ind['log'][token]]
        
        if table_name == 'map':
            return self.map[self.token2ind['map'][token]]
        
        if table_name == 'sample':
            return self.sample[self.token2ind['sample'][token]]
        
        if table_name == 'sample_annotation':
            return self.sample_annotation[self.token2ind['sample_annotation'][token]]
        
        if table_name == 'sample_data':
            return self.sample_data[self.token2ind['sample_data'][token]]
        
        if table_name == 'scene':
            return self.scene[self.token2ind['scene'][token]]
        
        if table_name == 'sensor':
            return self.sensor[self.token2ind['sensor'][token]]
        
        if table_name == 'visibility':
            return self.visibility[self.token2ind['visibility'][token]]
        


    def sample_decorate(self):

        # Decorate(add short-cut) sample_annotation table with for category_name 
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (add short-cut) sample_data with sensor information
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse index sample with sample_data and annotation 
        for record in self.sample:
            record['data'] = {}
            record['anns'] = [] 

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        
        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        


class Box:
   """ Simple data class representing a 3D box including label, score and velocity """     
   def __init__(self, center, size, orientation):
   
      assert type(center) == np.ndarray
      assert len(size) == 3
      assert type(orientation) == Quaternion
      
      
      self.center = center 
      self.size = size
      self.orientation = orientation
      
   def translate(self, x):
      
      self.center += x
      
   def rotate(self, orientation):
      self.center = np.dot(orientation.rotation_matrix, self.center)
      self.orientation = orientation * self.orientation  
      
           



def convert_kitti_bin_to_pcd(binFilePath):
    if binFilePath.endswith('.bin'):
    
       size_float = 4
       list_pcd = []
       with open(binFilePath, "rb") as f:
           byte = f.read(size_float * 5)
           while byte:
              x, y, z , intensity, ring_index = struct.unpack("fffff", byte)
              list_pcd.append([x, y, z])
              byte = f.read(size_float * 5)
       pcd = np.asarray(list_pcd)
       
       #pcd = o3d.geometry.PointCloud()
       #pcd.points = o3d.utility.Vector3dVector(np_pcd)
       return pcd






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
   
   


def quaternion_to_angle(quaternion):
    # Extract the scalar and vector components of the quaternion
    w, x, y, z = quaternion
    
    # Calculate the squared magnitude of the vector component
    magnitude_squared = x**2 + y**2 + z**2
    
    # If the magnitude is zero, return zero angle
    if magnitude_squared == 0:
        return 0.0
    
    # Calculate the angle using the arctangent function and return it in radians
    angle = 2.0 * np.arctan2(np.sqrt(magnitude_squared), w)
    return angle
    







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
   


   




   

   

   

   


   


   
   
   




class Data_nuScene(Dataset):
      def __init__(self, meta_path, data_path, train=True):
            self.meta_path = meta_path 
            self.data_path = data_path 
            self.train = train 

            self.nusc = Nuscene(self.meta_path)


            self.files = [] 
            
            frame_numbers = 0
            sample = {} 
            if self.train:
               self.scene = self.nusc.scene[0:750]
            else:
               self.scene = self.nusc.scene[750:]

            for i in range(len(self.scene)):
               my_scene = self.scene[i] 
               flag = 1
               first_flag = 1
               n_frame = 0 
   
               #print("scene table = ", my_scene) 
               instance_id = [] 
   
               while(flag==1):
                  n_frame = n_frame + 1 
                  frame_numbers = frame_numbers + 1 

                  if first_flag == 1:
                     first_sample_token = my_scene['first_sample_token']
                     last_sample_token = my_scene['last_sample_token'] 
   
                     my_sample = self.nusc.get('sample', first_sample_token) 
                     #print("sample table = ", my_sample)
                     next_sample = my_sample['next']
                     first_flag=0
                  else:
                     my_sample = self.nusc.get('sample', first_sample_token)
                     next_sample = my_sample['next']
         
                     #print("sample table = ", my_sample) 
   
      
      
                  # get sample lidar data
                  sensor = 'LIDAR_TOP'
                  lidar_data = self.nusc.get('sample_data', my_sample['data'][sensor]) 
                  #print("lidar data = ", lidar_data)
                  file_name = lidar_data['filename']
      
      
                  # get front camera data 
                  sensor = 'CAM_FRONT'
                  camera_data = self.nusc.get('sample_data', my_sample['data'][sensor])
                  cam_File = camera_data['filename']
      
      
      
                  file_path = os.path.join(data_path, file_name)
   
                  pcd = convert_kitti_bin_to_pcd(file_path)
      
                  # Retrive sensor and ego  pose record
                  cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                  sensor_record = self.nusc.get('sensor', cs_record['sensor_token'])
                  pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
      
      
      
                  annotation_token_list = my_sample['anns'] 
                  bb_list = []
                  for k in range(len(annotation_token_list)):
                     annotation_token = annotation_token_list[k] 
                     ann_record = self.nusc.get('sample_annotation', annotation_token) 
                     instance_id = ann_record['instance_token']
                     category = ann_record['category_name'] 
                     visibility_token = ann_record['visibility_token']
                     center = ann_record['translation']
                     size = ann_record['size']
                     rotation = ann_record['rotation']
                     box = Box(np.array(center), size, Quaternion(rotation))
         
         
         
         
                     # Move box to ego vehicle coordinate system
                     yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                     box.translate(-np.array(pose_record['translation']))
                     box.rotate(Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)]).inverse)
         
                     # Move box to sensor coordinate system 
                     box.translate(-np.array(cs_record['translation']))
                     #box.rotate(Quaternion(cs_record['rotation']).inverse)
                     yaw = Quaternion(cs_record['rotation']).yaw_pitch_roll[0]
                     box.rotate(Quaternion(scalar=np.cos(yaw/2), vector=[0,0,np.sin(yaw/2)]).inverse)
         
         
         
                     rotation_angle = quaternion_to_angle(box.orientation)
                     dim_of_box = [box.center, size[1], size[0], size[2], rotation_angle, category, instance_id]
                     bb_list.append(dim_of_box)
                  
         
                  #if n_frame == 10:
                  #obj.update_renderer(pcd, bb_list)
                  #time.sleep(0.5) 
                  if self.train:
                     if frame_numbers == 1:
                        sample["pcd_1"] = pcd 
                        sample["bb_list_1"] = bb_list 
                  
                     if frame_numbers == 2:
                        sample["pcd_2"] = pcd 
                        sample["bb_list_2"] = bb_list 

                     if frame_numbers == 3:
                        sample["pcd_3"] = pcd 
                        sample["bb_list_3"] = bb_list 

                     if frame_numbers == 4:
                        sample["pcd_4"] = pcd 
                        sample["bb_list_4"] = bb_list 

                     if frame_numbers == 5:
                        sample["pcd_5"] = pcd 
                        sample["bb_list_5"] = bb_list 

                     if frame_numbers == 6:
                        sample["pcd_6"] = pcd 
                        sample["bb_list_6"] = bb_list 

                     if frame_numbers == 7:
                        sample["pcd_7"] = pcd 
                        sample["bb_list_7"] = bb_list 

                     if frame_numbers == 8:
                        sample["pcd_8"] = pcd 
                        sample["bb_list_8"] = bb_list 

                     if frame_numbers == 9:
                        sample["pcd_9"] = pcd 
                        sample["bb_list_9"] = bb_list 

                     if frame_numbers == 10:
                        sample["pcd_10"] = pcd 
                        sample["bb_list_10"] = bb_list 

                     if frame_numbers == 11:
                        sample["pcd_11"] = pcd 
                        sample["bb_list_11"] = bb_list 

                     if frame_numbers == 12:
                        sample["pcd_12"] = pcd 
                        sample["bb_list_12"] = bb_list 

                     if frame_numbers == 13:
                        sample["pcd_13"] = pcd 
                        sample["bb_list_13"] = bb_list 

                     if frame_numbers == 14 :
                        sample["pcd_14"] = pcd 
                        sample["bb_list_14"] = bb_list 

                     if frame_numbers == 15:
                        sample["pcd_15"] = pcd 
                        sample["bb_list_15"] = bb_list 

                     if frame_numbers == 16:
                        sample["pcd_16"] = pcd 
                        sample["bb_list_16"] = bb_list 

                     if frame_numbers == 17 :
                        sample["pcd_17"] = pcd 
                        sample["bb_list_17"] = bb_list 

                     if frame_numbers == 18:
                        sample["pcd_18"] = pcd 
                        sample["bb_list_18"] = bb_list 

                     if frame_numbers == 19 :
                        sample["pcd_19"] = pcd 
                        sample["bb_list_19"] = bb_list 

                     if frame_numbers == 20 :
                        sample["pcd_20"] = pcd 
                        sample["bb_list_20"] = bb_list  

                  
                     if frame_numbers == 20 :
                        frame_numbers = 0
                        self.files.append(sample)
                        sample = {} 

                  else:
                     if frame_numbers == 1:
                        sample["pcd_1"] = pcd 
                        sample["bb_list_1"] = bb_list 

                     if frame_numbers == 2:
                        sample["pcd_2"] = pcd 
                        sample["bb_list_2"] = bb_list 

                     if frame_numbers == 2:
                        frame_numbers = 0 
                        self.files.append(sample)
                        sample = {} 
                     
                  first_sample_token = next_sample 
      
                  if next_sample == '':
                     flag=0 

      def __len__(self):
         return len(self.files)


      def __getitem__(self, index):
         if self.train:
            pcd_1 = self.files[index]["pcd_1"]
            pcd_2 = self.files[index]["pcd_2"]
            pcd_3 = self.files[index]["pcd_3"]
            pcd_4 = self.files[index]["pcd_4"]
            pcd_5 = self.files[index]["pcd_5"]
            pcd_6 = self.files[index]["pcd_6"]
            pcd_7 = self.files[index]["pcd_7"]
            pcd_8 = self.files[index]["pcd_8"]
            pcd_9 = self.files[index]["pcd_9"]
            pcd_10 = self.files[index]["pcd_10"]
            pcd_11 = self.files[index]["pcd_11"]
            pcd_12 = self.files[index]["pcd_12"]
            pcd_13 = self.files[index]["pcd_13"]
            pcd_14 = self.files[index]["pcd_14"]
            pcd_15 = self.files[index]["pcd_15"]
            pcd_16 = self.files[index]["pcd_16"]
            pcd_17 = self.files[index]["pcd_17"]
            pcd_18 = self.files[index]["pcd_18"]
            pcd_19 = self.files[index]["pcd_19"]
            pcd_20 = self.files[index]["pcd_20"]

            bb_list_1 = self.files[index]["bb_list_1"]
            bb_list_2 = self.files[index]["bb_list_2"]
            bb_list_3 = self.files[index]["bb_list_3"]
            bb_list_4 = self.files[index]["bb_list_4"]
            bb_list_5 = self.files[index]["bb_list_5"]
            bb_list_6 = self.files[index]["bb_list_6"]
            bb_list_7 = self.files[index]["bb_list_7"]
            bb_list_8 = self.files[index]["bb_list_8"]
            bb_list_9 = self.files[index]["bb_list_9"]
            bb_list_10 = self.files[index]["bb_list_10"]
            bb_list_11 = self.files[index]["bb_list_11"]
            bb_list_12 = self.files[index]["bb_list_12"]
            bb_list_13 = self.files[index]["bb_list_13"]
            bb_list_14 = self.files[index]["bb_list_14"]
            bb_list_15 = self.files[index]["bb_list_15"]
            bb_list_16 = self.files[index]["bb_list_16"]
            bb_list_17 = self.files[index]["bb_list_17"]
            bb_list_18 = self.files[index]["bb_list_18"]
            bb_list_19 = self.files[index]["bb_list_19"]
            bb_list_20 = self.files[index]["bb_list_20"]


            return {"pcd_1":pcd_1, "pcd_2":pcd_2, "pcd_3":pcd_3, "pcd_4":pcd_4, "pcd_5":pcd_5, "pcd_6":pcd_6, "pcd_7":pcd_7, "pcd_8":pcd_8,
                 "pcd_9":pcd_9, "pcd_10":pcd_10, "pcd_11":pcd_11, "pcd_12":pcd_12, "pcd_13":pcd_13, "pcd_14":pcd_14, "pcd_15":pcd_15 ,
                 "pcd_16":pcd_16, "pcd_17":pcd_17, "pcd_18":pcd_18, "pcd_19":pcd_19, "pcd_20":pcd_20 ,
                 "bb_list_1":bb_list_1, "bb_list_2":bb_list_2, "bb_list_3":bb_list_3, "bb_list_4":bb_list_4, "bb_list_5":bb_list_5 ,
                 "bb_list_6":bb_list_6, "bb_list_7":bb_list_7, "bb_list_8":bb_list_8, "bb_list_9":bb_list_9, "bb_list_10":bb_list_10 ,
                 "bb_list_11":bb_list_11, "bb_list_12":bb_list_12, "bb_list_13":bb_list_13, "bb_list_14":bb_list_14, "bb_list_15":bb_list_15 ,
                 "bb_list_16":bb_list_16, "bb_list_17":bb_list_17, "bb_list_18":bb_list_18, "bb_list_19":bb_list_19, "bb_list_20":bb_list_20} 
         else:
            pcd_1 = self.files[index]["pcd_1"] 
            pcd_2 = self.files[index]["pcd_2"]

            bb_list_1 = self.files[index]["bb_list_1"]
            bb_list_2 = self.files[index]["bb_list_2"]
            return {"pcd_1": pcd_1, "pcd_2":pcd_2, "bb_list_1":bb_list_1, "bb_list_2":bb_list_2}
         
         





class CustomData(Data):
   def __init__(self, velo_path, label_path):
      self.velo_path = velo_path 
      self.label_path = label_path 


      velo_pcds = sorted(os.listdir(self.velo_path))
      label_frames = sorted(os.listdir(label_path)) 

      
      self.files = []
      i = 0
      while i < len(velo_pcds)-1:
          pcd_file = velo_pcds[i]
          pcd_file_path = os.path.join(self.velo_path, pcd_file)
          # Load point cloud file
          pcd = o3d.io.read_point_cloud(pcd_file_path)
          points_1 = np.asarray(pcd.points)
        
          label = label_frames[i] 
          label_frame_path = os.path.join(label_path, label)
          label_file = open(label_frame_path, "r")
          label_f = label_file.read() 
          labels = json.loads(label_f) 
       
     
          sample = {}
          bboxes = [] 
          
          for l in range(len(labels)):
             obj_id = int(labels[l]["obj_id"]) 
             obj_type = labels[l]["obj_type"] 
             x = labels[l]["psr"]["position"]["x"]
             y = labels[l]["psr"]["position"]["y"] 
             z = labels[l]["psr"]["position"]["z"] 
             rot_x, rot_y, rot_z = labels[l]["psr"]["rotation"]["x"], labels[l]["psr"]["rotation"]["y"], labels[l]["psr"]["rotation"]["z"] 
              
             scale_l = labels[l]["psr"]["scale"]["x"]
             scale_w = labels[l]["psr"]["scale"]["y"]
             scale_h = labels[l]["psr"]["scale"]["z"] 
            
             if obj_type == "car":
                id = 0
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id])
             if obj_type == "Pedestrian":
                id = 1
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id]) 

             if obj_type == "Motorcycle":
                id = 2 
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id]) 
            

          pcd_file = velo_pcds[i+1]
          pcd_file_path = os.path.join(self.velo_path, pcd_file)
          # Load point cloud file
          pcd = o3d.io.read_point_cloud(pcd_file_path)
          points_2 = np.asarray(pcd.points)
          label = label_frames[i+1] 
          label_frame_path = os.path.join(label_path, label)
          label_file = open(label_frame_path, "r")
          label_f = label_file.read() 
          labels = json.loads(label_f) 
       
          sample["bboxes_1"] = bboxes 
          sample["points_1"] = points_1 

          
          bboxes = [] 
          
          for l in range(len(labels)):
             obj_id = int(labels[l]["obj_id"]) 
             obj_type = labels[l]["obj_type"] 
             x = labels[l]["psr"]["position"]["x"]
             y = labels[l]["psr"]["position"]["y"] 
             z = labels[l]["psr"]["position"]["z"] 
             rot_x, rot_y, rot_z = labels[l]["psr"]["rotation"]["x"], labels[l]["psr"]["rotation"]["y"], labels[l]["psr"]["rotation"]["z"] 
              
             scale_l = labels[l]["psr"]["scale"]["x"]
             scale_w = labels[l]["psr"]["scale"]["y"]
             scale_h = labels[l]["psr"]["scale"]["z"] 
            
             if obj_type == "car":
                id = 0
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id])
             if obj_type == "Pedestrian":
                id = 1
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id]) 

             if obj_type == "Motorcycle":
                id = 2 
                bboxes.append([x,y,z, scale_l, scale_w, scale_h, rot_z, id, obj_id])  

          sample["bboxes_2"] = bboxes
          sample["points_2"] = points_2
          i = i + 1 

         

         

          self.files.append(sample)

   def __len__(self):
      return len(self.files) 
   

   def __getitem__(self,index):
      bboxes_1 = torch.tensor(self.files[index]["bboxes_1"])
      bboxes_2 = torch.tensor(self.files[index]["bboxes_2"])
      points_1 = torch.tensor(self.files[index]["points_1"])
      points_2 = torch.tensor(self.files[index]["points_2"])

      return {"bboxes_1":bboxes_1, "bboxes_2":bboxes_2, "points_1":points_1, "points_2":points_2}
   

          
           

         

            
          
      

                     









         
         
   


   


   


   

   

   

   

   

   





   


   

   

   

   

   

   

      
      

   
   


   




   

   


           
             
             
             

     


     
