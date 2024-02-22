import torch 
import torch.nn as nn 
import numpy as np 
import json 
import os 
import open3d as o3d
import time
import matplotlib.pyplot as plt 
import struct 
from pyquaternion import Quaternion
import copy 



global lines 
lines = [] 
 

class NonBlockVisualizer_nuScene:
    def __init__(self, point_size=1, background_color=[0, 0, 0]):
        self.__visualizer = o3d.visualization.Visualizer()
        self.__visualizer.create_window()
        opt = self.__visualizer.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt = self.__visualizer.get_render_option()
        opt.point_size = point_size

        self.__pcd_vis = o3d.geometry.PointCloud()
        self.__initialized = False
        
       
        
        self.total_bboxes = []
        for i in range(500):
           self.total_bboxes.append(o3d.geometry.LineSet())
        
        self.count_bb = 0
     
         
        
        
        
        
    
    def add_bounding_box(self, center, length, width, height, orientation):
       # Define the vertices of the bounding box
       vertices = np.asarray([
             [length/2, width/2, height/2],
             [-length/2, width/2, height/2],
             [-length/2, -width/2, height/2],
             [length/2, -width/2, height/2],
             [length/2, width/2, -height/2],
             [-length/2, width/2, -height/2],
             [-length/2, -width/2, -height/2],
             [length/2, -width/2, -height/2]])

       # Rotate the vertices based on the orientation
       R = np.array([[np.cos(orientation), -np.sin(orientation), 0],
                  [np.sin(orientation), np.cos(orientation), 0],
                  [0, 0, 1]])
       vertices = vertices @ R.T

       # Translate the vertices to the center of the bounding box
       vertices = vertices + center

       # Define the edges of the bounding box
       edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])

       # Create the LineSet geometry
       #line_set = o3d.geometry.LineSet()
       #line_set.points = o3d.utility.Vector3dVector(vertices)
       #line_set.lines = o3d.utility.Vector2iVector(edges)

       # Set the colors of the edges
       #colors = [[1, 0, 0] for i in range(len(edges))]
       #line_set.colors = o3d.utility.Vector3dVector(colors)
       
       
       
 
       
       
       return vertices, edges
       
       
    

    def update_renderer(self, pcd, bb_list, wait_time=0):
        self.__pcd_vis.points = pcd.points
        self.__pcd_vis.colors = pcd.colors
        
        
        if not self.__initialized:
            self.__initialized = True
            self.__visualizer.add_geometry(self.__pcd_vis)
            
            for dim_of_bb in bb_list:
               vertices, edges = self.add_bounding_box(dim_of_bb[0], dim_of_bb[1], dim_of_bb[2], dim_of_bb[3], dim_of_bb[4]) 
               track_id = dim_of_bb[5] 
        
               #self.__visualizer.add_geometry(line_set)
               #lines.append(line_set)
               self.total_bboxes[self.count_bb].points = o3d.utility.Vector3dVector(vertices)
               self.total_bboxes[self.count_bb].lines = o3d.utility.Vector2iVector(edges) 
               
               if track_id == 101:
                  colors = [[0, 0, 0.2] for i in range(len(edges))] 
                
               if  track_id == 100:
                  colors = [[0, 0, 0.4] for i in range(len(edges))] 
                  
               if track_id == 99:
                  colors = [[0, 0, 0.6] for i in range(len(edges))] 
                  
               if track_id == 98:
                  colors = [[0, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 97:
                  colors = [[0, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 96:
                  colors = [[0, 0.2, 0.2] for i in range(len(edges))] 
               
               if track_id == 95:
                  colors = [[0, 0.2, 0.4] for i in range(len(edges))] 
                  
               if track_id == 94:
                  colors = [[0, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 93:
                  colors = [[0, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 92:
                  colors = [[0, 0.2, 1] for i in range(len(edges))] 
                  
               if track_id == 91:
                  colors = [[0, 0.4, 0.2] for i in range(len(edges))]
                  
               if track_id == 90:
                  colors = [[0, 0.4, 0.4] for i in range(len(edges))]
                  
               if track_id == 89:
                  colors = [[0, 0.4, 0.6] for i in range(len(edges))] 
                  
               if track_id == 88:
                  colors = [[0, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 87:
                  colors = [[0, 0.4, 1] for i in range(len(edges))] 
                  
               if track_id == 86:
                  colors = [[0, 0.6, 0.2] for i in range(len(edges))] 
                  
               if track_id == 85:
                  colors = [[0, 0.6, 0.4] for i in range(len(edges))] 
                  
               if track_id == 84:
                  colors = [[0, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 83:
                  colors = [[0, 0.6, 1] for i in range(len(edges))] 
                  
               if track_id == 82:
                  colors = [[0, 0.8, 0.2] for i in range(len(edges))] 
                  
               if track_id == 81:
                  colors = [[0, 0.8, 0.4] for i in range(len(edges))]
                  
               if track_id == 80:
                  colors = [[0, 0.8, 0.6] for i in range(len(edges))] 
                  
               if track_id == 79:
                  colors = [[0, 0.8, 0.8] for i in range(len(edges))] 
                  
               if track_id == 78:
                  colors = [[0, 0.8, 1] for i in range(len(edges))] 
                  
               if track_id == 77:
                  colors = [[0, 1, 0] for i in range(len(edges))] 
                  
               if track_id == 76:
                  colors = [[0, 1, 0.2] for i in range(len(edges))] 
                  
               if track_id == 75 :
                  colors = [[0, 1, 0.4] for i in range(len(edges))] 
                  
               if track_id == 74 :
                  colors = [[0, 1, 0.6] for i in range(len(edges))] 
                  
               if track_id == 73 :
                  colors = [[0, 1, 0.8] for i in range(len(edges))] 
                  
               if track_id == 72 :
                  colors = [[0, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 71:
                  colors = [[0.2, 0, 0] for i in range(len(edges))] 
               
               if track_id == 70 :
                  colors = [[0.2, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 69:
                  colors = [[0.2, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 68 :
                  colors = [[0.2, 0, 0.6] for i in range(len(edges))] 
                  
               if track_id == 67 :
                  colors = [[0.2, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 66 :
                  colors = [[0.2, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 65:
                  colors = [[0.2, 0.2, 0] for i in range(len(edges))]
                  
               if track_id == 64 :
                  colors = [[0.2, 0.2, 0.2] for i in range(len(edges))] 
                  
               if track_id == 63 :
                  colors = [[0.2, 0.2, 0.4] for i in range(len(edges))]
                  
               
               if track_id == 62:
                  colors = [[0.2, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 61:
                  colors = [[0.2, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 60:
                  colors = [[0.2, 0.2, 1] for i in range(len(edges))]
                  
               if track_id == 59:
                  colors = [[0.2, 0.4, 0] for i in range(len(edges))] 
                  
               if track_id == 58:
                  colors = [[0.2, 0.4, 0.2] for i in range(len(edges))] 
                  
               if track_id == 57:
                  colors = [[0.2, 0.4, 0.4] for i in range(len(edges))]
                  
               if track_id == 56:
                  colors = [[0.2, 0.4, 0.6] for i in range(len(edges))]
                  
               if track_id == 55:
                  colors = [[0.2, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 54:
                  colors = [[0.2, 0.4, 1] for i in range(len(edges))]
                  
               if track_id == 53:
                  colors = [[0.2, 0.6, 0] for i in range(len(edges))]
                  
               if track_id == 52:
                  colors = [[0.2, 0.6, 0.2] for i in range(len(edges))]
                  
               if track_id == 51:
                  colors = [[0.2, 0.6, 0.4] for i in range(len(edges))]
                  
               if track_id == 50:
                  colros = [[0.2, 0.6, 0.6] for i in range(len(edges))]
                  
               if track_id == 49:
                  colors = [[0.2, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 48:
                  colors = [[0.2, 0.6, 1] for i in range(len(edges))]
                  
               if track_id == 47:
                  colors = [[0.2, 0.8, 0] for i in range(len(edges))]
                  
               if track_id == 46:
                  colors = [[0.2, 1, 0] for i in range(len(edges))]
                  
               if track_id == 45:
                  colors = [[0.2, 1, 0.2] for i in range(len(edges))]
                  
               if track_id == 44:
                  colors = [[0.2, 1, 0.4] for i in range(len(edges))] 
                  
               if track_id == 43:
                  colors = [[0.2, 1, 0.6] for i in range(len(edges))]
                  
               if track_id == 42:
                  colors = [[0.2, 1, 0.8] for i in range(len(edges))] 
                  
               if track_id == 41:
                  colors = [[0.2, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 40:
                  colors = [[0.4, 0, 0] for i in range(len(edges))]
                  
               if track_id == 39:
                  colors = [[0.4, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 38:
                  colors = [[0.4, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 37:
                  colors = [[0.4, 0, 0.6] for i in range(len(edges))]
                  
               if track_id == 36:
                  colors = [[0.4, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 35:
                  colors = [[0.4, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 34:
                  colors = [[0.4, 0.2, 0] for i in range(len(edges))]
                  
               if track_id == 33:
                  colors = [[0.4, 0.2, 0.2] for i in range(len(edges))]
                  
               if track_id == 32:
                  colors = [[0.4, 0.2, 0.4] for i in range(len(edges))]
                  
               if track_id == 31:
                  colors = [[0.4, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 30:
                  colors = [[0.4, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 29:
                  colors = [[0.4, 0.2, 1] for i in range(len(edges))] 
                  
               if track_id == 28:
                  colors = [[0.4, 0.4, 0] for i in range(len(edges))] 
                  
               if track_id == 27:
                  colors = [[0.4, 0.4, 0.2] for i in range(len(edges))] 
                  
               if track_id == 26:
                  colors = [[0.4, 0.4, 0.4] for i in range(len(edges))] 
                  
               if track_id == 25:
                  colors = [[0.4, 0.4, 0.6] for i in range(len(edges))]
                  
               if track_id == 24:
                  colors = [[0.4, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 23:
                  colors = [[0.4, 0.4, 1] for i in range(len(edges))]
                  
               if track_id == 22:
                  colors = [[0.4, 0.6, 0] for i in range(len(edges))]
                  
               if track_id == 21:
                  colors = [[0.4, 0.6, 0.2] for i in range(len(edges))]
                  
               if track_id == 20:
                  colors = [[0.4, 0.6, 0.4] for i in range(len(edges))] 
                  
               if track_id == 19:
                  colors = [[0.4, 0.6, 0.6] for i in range(len(edges))]
                  
               if track_id == 18:
                  colors = [[0.4, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 17:
                  colors = [[0.4, 0.6, 1] for i in range(len(edges))]
                  
               if track_id == 16:
                  colors = [[0.4, 0.8, 0] for i in range(len(edges))]
                  
               if track_id == 15:
                  colors = [[0.4, 0.8, 0.2] for i in range(len(edges))]
                  
               if track_id == 14:
                  colors = [[0.4, 0.8, 0.4] for i in range(len(edges))]
                  
               if track_id == 13:
                  colors = [[0.4, 0.8, 0.6] for i in range(len(edges))] 
                  
               if track_id == 12:
                  colors = [[0.4, 0.8, 0.8] for i in range(len(edges))]
                  
               if track_id == 11:
                  colors = [[0.4, 0.8, 1] for i in range(len(edges))]
                  
               if track_id == 10:
                  colors = [[0.4, 1, 0] for i in range(len(edges))]
                  
               if track_id == 9:
                  colors = [[0.4, 1, 0.2] for i in range(len(edges))] 
                  
               if track_id == 8:
                  colors = [[0.4, 1, 0.4] for i in range(len(edges))]
                  
               if track_id == 7:
                  colors = [[0.4, 1, 0.6] for i in range(len(edges))]
                  
               if track_id == 6:
                  colors = [[0.4, 1, 0.8] for i in range(len(edges))]
                  
               if track_id == 5:
                  colors = [[0.4, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 4:
                  colors = [[0.6, 0, 0] for i in range(len(edges))]
                  
               if track_id == 3:
                  colors = [[0.6, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 2:
                  colors = [[0.6, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 1:
                  colors = [[0.6, 0, 0.6] for i in range(len(edges))]
                  
               if track_id == 0:
                  colors = [[0.6, 0, 0.8] for i in range(len(edges))]
                  
                  
             
                 
                        
                  
                  
               self.total_bboxes[self.count_bb].colors = o3d.utility.Vector3dVector(colors) 
               self.count_bb = self.count_bb + 1
             
            for bbox in self.total_bboxes:
               self.__visualizer.add_geometry(bbox)
               
            
               
               
                  
        else:
            self.__visualizer.update_geometry(self.__pcd_vis)
            
            for i in range(self.count_bb):
               self.total_bboxes[i].points = o3d.utility.Vector3dVector([])
               self.total_bboxes[i].lines = o3d.utility.Vector2iVector([]) 
               self.total_bboxes[i].colors = o3d.utility.Vector3dVector([])
               
            self.count_bb = 0
            for dim_of_bb in bb_list:
               vertices, edges = self.add_bounding_box(dim_of_bb[0], dim_of_bb[1], dim_of_bb[2], dim_of_bb[3], dim_of_bb[4]) 
               track_id = dim_of_bb[5] 
               
               
               self.total_bboxes[self.count_bb].points = o3d.utility.Vector3dVector(vertices)
               self.total_bboxes[self.count_bb].lines = o3d.utility.Vector2iVector(edges) 
               
               if track_id == 101:
                  colors = [[0, 0, 0.2] for i in range(len(edges))] 
                
               if  track_id == 100:
                  colors = [[0, 0, 0.4] for i in range(len(edges))] 
                  
               if track_id == 99:
                  colors = [[0, 0, 0.6] for i in range(len(edges))] 
                  
               if track_id == 98:
                  colors = [[0, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 97 :
                  colors = [[0, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 96:
                  colors = [[0, 0.2, 0.2] for i in range(len(edges))] 
               
               if track_id == 95:
                  colors = [[0, 0.2, 0.4] for i in range(len(edges))] 
                  
               if track_id == 94:
                  colors = [[0, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 93:
                  colors = [[0, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 92:
                  colors = [[0, 0.2, 1] for i in range(len(edges))] 
                  
               if track_id == 91:
                  colors = [[0, 0.4, 0.2] for i in range(len(edges))]
                  
               if track_id == 90:
                  colors = [[0, 0.4, 0.4] for i in range(len(edges))]
                  
               if track_id == 89:
                  colors = [[0, 0.4, 0.6] for i in range(len(edges))] 
                  
               if track_id == 88:
                  colors = [[0, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 87:
                  colors = [[0, 0.4, 1] for i in range(len(edges))] 
                  
               if track_id == 86:
                  colors = [[0, 0.6, 0.2] for i in range(len(edges))] 
                  
               if track_id == 85:
                  colors = [[0, 0.6, 0.4] for i in range(len(edges))] 
                  
               if track_id == 84:
                  colors = [[0, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 83:
                  colors = [[0, 0.6, 1] for i in range(len(edges))] 
                  
               if track_id == 82:
                  colors = [[0, 0.8, 0.2] for i in range(len(edges))] 
                  
               if track_id == 81:
                  colors = [[0, 0.8, 0.4] for i in range(len(edges))]
                  
               if track_id == 80:
                  colors = [[0, 0.8, 0.6] for i in range(len(edges))] 
                  
               if track_id == 79:
                  colors = [[0, 0.8, 0.8] for i in range(len(edges))] 
                  
               if track_id == 78:
                  colors = [[0, 0.8, 1] for i in range(len(edges))] 
                  
               if track_id == 77:
                  colors = [[0, 1, 0] for i in range(len(edges))] 
                  
               if track_id == 76:
                  colors = [[0, 1, 0.2] for i in range(len(edges))] 
                  
               if track_id == 75 :
                  colors = [[0, 1, 0.4] for i in range(len(edges))] 
                  
               if track_id == 74 :
                  colors = [[0, 1, 0.6] for i in range(len(edges))] 
                  
               if track_id == 73 :
                  colors = [[0, 1, 0.8] for i in range(len(edges))] 
                  
               if track_id == 72 :
                  colors = [[0, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 71:
                  colors = [[0.2, 0, 0] for i in range(len(edges))] 
               
               if track_id == 70 :
                  colors = [[0.2, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 69:
                  colors = [[0.2, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 68 :
                  colors = [[0.2, 0, 0.6] for i in range(len(edges))] 
                  
               if track_id == 67 :
                  colors = [[0.2, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 66 :
                  colors = [[0.2, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 65:
                  colors = [[0.2, 0.2, 0] for i in range(len(edges))]
                  
               if track_id == 64 :
                  colors = [[0.2, 0.2, 0.2] for i in range(len(edges))] 
                  
               if track_id == 63 :
                  colors = [[0.2, 0.2, 0.4] for i in range(len(edges))]
                  
               
               if track_id == 62:
                  colors = [[0.2, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 61:
                  colors = [[0.2, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 60:
                  colors = [[0.2, 0.2, 1] for i in range(len(edges))]
                  
               if track_id == 59:
                  colors = [[0.2, 0.4, 0] for i in range(len(edges))] 
                  
               if track_id == 58:
                  colors = [[0.2, 0.4, 0.2] for i in range(len(edges))] 
                  
               if track_id == 57:
                  colors = [[0.2, 0.4, 0.4] for i in range(len(edges))]
                  
               if track_id == 56:
                  colors = [[0.2, 0.4, 0.6] for i in range(len(edges))]
                  
               if track_id == 55:
                  colors = [[0.2, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 54:
                  colors = [[0.2, 0.4, 1] for i in range(len(edges))]
                  
               if track_id == 53:
                  colors = [[0.2, 0.6, 0] for i in range(len(edges))]
                  
               if track_id == 52:
                  colors = [[0.2, 0.6, 0.2] for i in range(len(edges))]
                  
               if track_id == 51:
                  colors = [[0.2, 0.6, 0.4] for i in range(len(edges))]
                  
               if track_id == 50:
                  colros = [[0.2, 0.6, 0.6] for i in range(len(edges))]
                  
               if track_id == 49:
                  colors = [[0.2, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 48:
                  colors = [[0.2, 0.6, 1] for i in range(len(edges))]
                  
               if track_id == 47:
                  colors = [[0.2, 0.8, 0] for i in range(len(edges))]
                  
               if track_id == 46:
                  colors = [[0.2, 1, 0] for i in range(len(edges))]
                  
               if track_id == 45:
                  colors = [[0.2, 1, 0.2] for i in range(len(edges))]
                  
               if track_id == 44:
                  colors = [[0.2, 1, 0.4] for i in range(len(edges))] 
                  
               if track_id == 43:
                  colors = [[0.2, 1, 0.6] for i in range(len(edges))]
                  
               if track_id == 42:
                  colors = [[0.2, 1, 0.8] for i in range(len(edges))] 
                  
               if track_id == 41:
                  colors = [[0.2, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 40:
                  colors = [[0.4, 0, 0] for i in range(len(edges))]
                  
               if track_id == 39:
                  colors = [[0.4, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 38:
                  colors = [[0.4, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 37:
                  colors = [[0.4, 0, 0.6] for i in range(len(edges))]
                  
               if track_id == 36:
                  colors = [[0.4, 0, 0.8] for i in range(len(edges))] 
                  
               if track_id == 35:
                  colors = [[0.4, 0, 1] for i in range(len(edges))] 
                  
               if track_id == 34:
                  colors = [[0.4, 0.2, 0] for i in range(len(edges))]
                  
               if track_id == 33:
                  colors = [[0.4, 0.2, 0.2] for i in range(len(edges))]
                  
               if track_id == 32:
                  colors = [[0.4, 0.2, 0.4] for i in range(len(edges))]
                  
               if track_id == 31:
                  colors = [[0.4, 0.2, 0.6] for i in range(len(edges))]
                  
               if track_id == 30:
                  colors = [[0.4, 0.2, 0.8] for i in range(len(edges))]
                  
               if track_id == 29:
                  colors = [[0.4, 0.2, 1] for i in range(len(edges))] 
                  
               if track_id == 28:
                  colors = [[0.4, 0.4, 0] for i in range(len(edges))] 
                  
               if track_id == 27:
                  colors = [[0.4, 0.4, 0.2] for i in range(len(edges))] 
                  
               if track_id == 26:
                  colors = [[0.4, 0.4, 0.4] for i in range(len(edges))] 
                  
               if track_id == 25:
                  colors = [[0.4, 0.4, 0.6] for i in range(len(edges))]
                  
               if track_id == 24:
                  colors = [[0.4, 0.4, 0.8] for i in range(len(edges))] 
                  
               if track_id == 23:
                  colors = [[0.4, 0.4, 1] for i in range(len(edges))]
                  
               if track_id == 22:
                  colors = [[0.4, 0.6, 0] for i in range(len(edges))]
                  
               if track_id == 21:
                  colors = [[0.4, 0.6, 0.2] for i in range(len(edges))]
                  
               if track_id == 20:
                  colors = [[0.4, 0.6, 0.4] for i in range(len(edges))] 
                  
               if track_id == 19:
                  colors = [[0.4, 0.6, 0.6] for i in range(len(edges))]
                  
               if track_id == 18:
                  colors = [[0.4, 0.6, 0.8] for i in range(len(edges))] 
                  
               if track_id == 17:
                  colors = [[0.4, 0.6, 1] for i in range(len(edges))]
                  
               if track_id == 16:
                  colors = [[0.4, 0.8, 0] for i in range(len(edges))]
                  
               if track_id == 15:
                  colors = [[0.4, 0.8, 0.2] for i in range(len(edges))]
                  
               if track_id == 14:
                  colors = [[0.4, 0.8, 0.4] for i in range(len(edges))]
                  
               if track_id == 13:
                  colors = [[0.4, 0.8, 0.6] for i in range(len(edges))] 
                  
               if track_id == 12:
                  colors = [[0.4, 0.8, 0.8] for i in range(len(edges))]
                  
               if track_id == 11:
                  colors = [[0.4, 0.8, 1] for i in range(len(edges))]
                  
               if track_id == 10:
                  colors = [[0.4, 1, 0] for i in range(len(edges))]
                  
               if track_id == 9:
                  colors = [[0.4, 1, 0.2] for i in range(len(edges))] 
                  
               if track_id == 8:
                  colors = [[0.4, 1, 0.4] for i in range(len(edges))]
                  
               if track_id == 7:
                  colors = [[0.4, 1, 0.6] for i in range(len(edges))]
                  
               if track_id == 6:
                  colors = [[0.4, 1, 0.8] for i in range(len(edges))]
                  
               if track_id == 5:
                  colors = [[0.4, 1, 1] for i in range(len(edges))] 
                  
               if track_id == 4:
                  colors = [[0.6, 0, 0] for i in range(len(edges))]
                  
               if track_id == 3:
                  colors = [[0.6, 0, 0.2] for i in range(len(edges))]
                  
               if track_id == 2:
                  colors = [[0.6, 0, 0.4] for i in range(len(edges))]
                  
               if track_id == 1:
                  colors = [[0.6, 0, 0.6] for i in range(len(edges))]
                  
               if track_id == 0:
                  colors = [[0.6, 0, 0.8] for i in range(len(edges))]
                  
                  
             
                 
               
               self.total_bboxes[self.count_bb].colors = o3d.utility.Vector3dVector(colors) 
               self.count_bb = self.count_bb + 1
               
               
            for bbox in self.total_bboxes:
               self.__visualizer.update_geometry(bbox)
               
               
               
               
            
            """for i in range(len(lines)):
               self.__visualizer.remove_geometry(lines[i])
               
            lines.clear() 
          
            for dim_of_bb in bb_list:
               line_set = self.add_bounding_box(dim_of_bb[0], dim_of_bb[1], dim_of_bb[2], dim_of_bb[3], dim_of_bb[4], )
               self.__visualizer.add_geometry(line_set)
               lines.append(line_set) """
        
          
        
           
        
        
                 
        
       
        self.__visualizer.poll_events()
        self.__visualizer.update_renderer()
        
        
        #self.__visualizer.run() 
        
        




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
       np_pcd = np.asarray(list_pcd)
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(np_pcd)
       return pcd









         


       