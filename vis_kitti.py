import numpy as np 
import open3d as o3d 
import os 
import struct 
import time 







global edges 

edges = np.array([[0, 1], [1,2], [2,3], [3,0],
                 [4,5], [5,6], [6, 7], [7,4],
                 [0,4], [1,5], [2,6], [3, 7]])

class NonBlockVisualizer:
    def __init__(self, point_size=2, background_color=[0, 0, 0]):
        self.__visualizer = o3d.visualization.Visualizer()
        
        self.__visualizer.create_window()
        opt = self.__visualizer.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt = self.__visualizer.get_render_option()
        opt.point_size = point_size

        self.__pcd_vis = o3d.geometry.PointCloud()
        self.__initialized = False
        
        self.total_bboxes = [] 
        self.total_bb_text = []
        for i in range(100):
           self.total_bboxes.append(o3d.geometry.LineSet())
              
        self.count_bb = 0 
        
        

    def update_renderer(self, pcd, bboxes , wait_time=0):
        self.__pcd_vis.points = pcd.points
        self.__pcd_vis.colors = pcd.colors

        if not self.__initialized:
            self.__initialized = True
            self.__visualizer.add_geometry(self.__pcd_vis)
            
            text_labels = []
            
            for dim_of_bb in bboxes:
               vertices = dim_of_bb[1] 
               category = dim_of_bb[2] 
               
               #print("category = ", category)
               self.total_bboxes[self.count_bb].points = o3d.utility.Vector3dVector(vertices)
               self.total_bboxes[self.count_bb].lines = o3d.utility.Vector2iVector(edges) 

               if category == 0:
                  colors = [[0.3,0,0] for i in range(len(edges))] 
                     
               if category == 1:
                  colors = [[0.6,0,0] for i in range(len(edges))] 
                    
               if category == 2:
                  colors = [[1,0,0] for i in range(len(edges))] 
                      
               if category == 3:
                  colors = [[0,0.3,0]for i in range(len(edges))] 
                     
               if category == 4 :
                  colors = [[0,0.6,0] for i in range(len(edges))] 
                     
               if category == 5:
                  colors = [[0, 1, 0] for i in range(len(edges))] 
                    
               if category == 6:
                  colors = [[0, 0, 0.3] for i in range(len(edges))] 
                    
               if category == 7:
                  colors = [[0,0, 0.6] for i in range(len(edges))]  
                      
               if category == 8:
                  colors = [[0, 0, 1] for i in range(len(edges))] 
                     
               if category == 9:
                  colors = [[1, 0.3, 0] for i  in range(len(edges))]  
               
               if category == 10:
                  colors = [[1, 0.6, 0] for i in range(len(edges))]

               if category == 11:
                  colors = [[1, 1, 0] for i in range(len(edges))]

               if category == 12 :
                  colors = [[1, 1, 0.3] for i in range(len(edges))]

               if category == 13 :
                  colors = [[1, 1, 0.6] for i in range(len(edges))]

               if category == 14 :
                  colors = [[1, 1, 1] for i in range(len(edges))]

               if category == 15 :
                  colors = [[0.2, 0, 0] for i in range(len(edges))]

               if category == 16 :
                  colors = [[0.4, 0, 0] for i in range(len(edges))]

               if category == 17 :
                  colors = [[0.8, 0, 0] for i in range(len(edges))] 

               if category == 18 :
                  colors = [[0.8, 0.2, 0] for i in range(len(edges))] 

               if category == 19 :
                   colors = [[0.8, 0.4, 0] for i in range(len(edges))]

               if category == 20 :
                  colors = [[0.8, 0.8, 0] for i in range(len(edges))] 

               if category == 21 :
                  colors = [[0.8, 0.8, 0.2] for i in range(len(edges))]

               if category == 22 :
                  colors = [[0.8, 0.8, 0.4] for i in range(len(edges))]

               if category == 23 :
                  colors = [[0.8, 0.8, 0.8] for i in range(len(edges))] 

               if category == 24 :
                  colors = [[0.2, 0.2, 0] for i  in range(len(edges))]

               if category == 25 :
                  colors = [[0.2, 0.4, 0] for i in range(len(edges))] 

               if category == 26 :
                  colors = [[0.2, 0.6, 0] for i in range(len(edges))] 

               if category == 27 :
                  colors = [[0.2, 0.8, 0] for i in range(len(edges))] 

               if category == 28 :
                  colors = [[0.2, 1, 0] for i in range(len(edges))] 

               if category == 29 :
                  colors = [[0.2, 0.2, 0.2] for i in range(len(edges))] 

               if category == 30 :
                  colors = [[0.2, 0.2, 0.4] for i in range(len(edges))]  

               if category == 31 :
                  colors = [[0.2, 0.2, 0.6] for i in range(len(edges))]
               
               if category == 32 :
                  colors = [[0.2, 0.2, 0.8] for i in range(len(edges))] 

               if category == 33 :
                  colors = [[0.2, 0.2, 1] for i in range(len(edges))] 

               if category == 34 :
                  colors = [[0.2, 0.4, 0.2] for i in range(len(edges))] 

               if category == 35 :
                  colors = [[0.2, 0.4, 0.4] for i in range(len(edges))] 

               if category == 36 :
                  colors = [[0.2, 0.4, 0.6] for i in range(len(edges))] 

               if category == 37 :
                  colors = [[0.2, 0.4, 0.8] for i in range(len(edges))] 

               if category == 38 :
                  colors = [[0.2, 0.4, 1] for i in range(len(edges))] 

               if category == 39 :
                  colors = [[0.2, 0.6, 0.2] for i in range(len(edges))] 

               if category == 40:
                  colors = [[0.2, 0.6, 0.4] for i in range(len(edges))] 
                  

               if category == 41 :
                  colors = [[0.2, 0.6, 1] for i in range(len(edges))] 

               if category == 42 :
                  colors = [[0.2, 0.8, 0.2] for i in range(len(edges))] 

               if category == 43 :
                  colors = [[0.2, 0.8, 0.4] for i in range(len(edges))] 

               if category == 44 :
                  colors = [[0.2, 0.8, 0.6] for i in range(len(edges))] 

               if category == 45 :
                  colors = [[0.2, 0.8, 1] for i in range(len(edges))] 

               if category == 46 :
                  colors = [[0.4, 0.2, 0 ] for i in range(len(edges))]  

               if category == 47 :
                  colors = [[0.4, 0.2, 0.2] for i in range(len(edges))] 
               
               if category == 48 :
                  colors = [[0.4, 0.2, 0.6] for i in range(len(edges))] 

               if category == 49 :
                  colors = [[0.4, 0.2, 0.8] for i in range(len(edges))] 

               if category == 50 :
                  colors = [[0.4, 0.4, 0.2] for i in range(len(edges))] 
                  
                   
                  
           
                  
              
               self.total_bboxes[self.count_bb].colors = o3d.utility.Vector3dVector(colors) 
               self.count_bb = self.count_bb + 1 
               
            for bbox in self.total_bboxes:
               self.__visualizer.add_geometry(bbox) 
            
            # remove existing text labels from the visualizer    
            for label in text_labels:
               self.__visualizer.remove_geometry(label) 
                
               
                 
        else:
            self.__visualizer.update_geometry(self.__pcd_vis)
            
            for i in range(self.count_bb):
               self.total_bboxes[i].points = o3d.utility.Vector3dVector([]) 
               self.total_bboxes[i].lines = o3d.utility.Vector2iVector([]) 
               self.total_bboxes[i].colors = o3d.utility.Vector3dVector([]) 
               
            self.count_bb = 0 
            
            text_labels = [] 
            
            for dim_of_bb in bboxes:
               vertices = dim_of_bb[1] 
               category = dim_of_bb[2]
               print("category = ", category)
               self.total_bboxes[self.count_bb].points = o3d.utility.Vector3dVector(vertices) 
               self.total_bboxes[self.count_bb].lines = o3d.utility.Vector2iVector(edges) 
               
               if category == 0:
                  colors = [[0.3,0,0] for i in range(len(edges))] 
                     
               if category == 1:
                  colors = [[0.6,0,0] for i in range(len(edges))] 
                    
               if category == 2:
                  colors = [[1,0,0] for i in range(len(edges))] 
                      
               if category == 3:
                  colors = [[0,0.3,0]for i in range(len(edges))] 
                     
               if category == 4 :
                  colors = [[0,0.6,0] for i in range(len(edges))] 
                     
               if category == 5:
                  colors = [[0, 1, 0] for i in range(len(edges))] 
                    
               if category == 6:
                  colors = [[0, 0, 0.3] for i in range(len(edges))] 
                    
               if category == 7:
                  colors = [[0,0, 0.6] for i in range(len(edges))]  
                      
               if category == 8:
                  colors = [[0, 0, 1] for i in range(len(edges))] 
                     
               if category == 9:
                  colors = [[1, 0.3, 0] for i  in range(len(edges))]  
               
               if category == 10:
                  colors = [[1, 0.6, 0] for i in range(len(edges))]

               if category == 11:
                  colors = [[1, 1, 0] for i in range(len(edges))]

               if category == 12 :
                  colors = [[1, 1, 0.3] for i in range(len(edges))]

               if category == 13 :
                  colors = [[1, 1, 0.6] for i in range(len(edges))]

               if category == 14 :
                  colors = [[1, 1, 1] for i in range(len(edges))]

               if category == 15 :
                  colors = [[0.2, 0, 0] for i in range(len(edges))]

               if category == 16 :
                  colors = [[0.4, 0, 0] for i in range(len(edges))]

               if category == 17 :
                  colors = [[0.8, 0, 0] for i in range(len(edges))] 

               if category == 18 :
                  colors = [[0.8, 0.2, 0] for i in range(len(edges))] 

               if category == 19 :
                   colors = [[0.8, 0.4, 0] for i in range(len(edges))]

               if category == 20 :
                  colors = [[0.8, 0.8, 0] for i in range(len(edges))] 

               if category == 21 :
                  colors = [[0.8, 0.8, 0.2] for i in range(len(edges))]

               if category == 22 :
                  colors = [[0.8, 0.8, 0.4] for i in range(len(edges))]

               if category == 23 :
                  colors = [[0.8, 0.8, 0.8] for i in range(len(edges))] 

               if category == 24 :
                  colors = [[0.2, 0.2, 0] for i  in range(len(edges))]

               if category == 25 :
                  colors = [[0.2, 0.4, 0] for i in range(len(edges))] 

               if category == 26 :
                  colors = [[0.2, 0.6, 0] for i in range(len(edges))] 

               if category == 27 :
                  colors = [[0.2, 0.8, 0] for i in range(len(edges))] 

               if category == 28 :
                  colors = [[0.2, 1, 0] for i in range(len(edges))] 

               if category == 29 :
                  colors = [[0.2, 0.2, 0.2] for i in range(len(edges))] 

               if category == 30 :
                  colors = [[0.2, 0.2, 0.4] for i in range(len(edges))]  

               if category == 31 :
                  colors = [[0.2, 0.2, 0.6] for i in range(len(edges))]
               
               if category == 32 :
                  colors = [[0.2, 0.2, 0.8] for i in range(len(edges))] 

               if category == 33 :
                  colors = [[0.2, 0.2, 1] for i in range(len(edges))] 

               if category == 34 :
                  colors = [[0.2, 0.4, 0.2] for i in range(len(edges))] 

               if category == 35 :
                  colors = [[0.2, 0.4, 0.4] for i in range(len(edges))] 

               if category == 36 :
                  colors = [[0.2, 0.4, 0.6] for i in range(len(edges))] 

               if category == 37 :
                  colors = [[0.2, 0.4, 0.8] for i in range(len(edges))] 

               if category == 38 :
                  colors = [[0.2, 0.4, 1] for i in range(len(edges))] 

               if category == 39 :
                  colors = [[0.2, 0.6, 0.2] for i in range(len(edges))] 

               if category == 40:
                  colors = [[0.2, 0.6, 0.4] for i in range(len(edges))] 
                  

               if category == 41 :
                  colors = [[0.2, 0.6, 1] for i in range(len(edges))] 

               if category == 42 :
                  colors = [[0.2, 0.8, 0.2] for i in range(len(edges))] 

               if category == 43 :
                  colors = [[0.2, 0.8, 0.4] for i in range(len(edges))] 

               if category == 44 :
                  colors = [[0.2, 0.8, 0.6] for i in range(len(edges))] 

               if category == 45 :
                  colors = [[0.2, 0.8, 1] for i in range(len(edges))] 

               if category == 46 :
                  colors = [[0.4, 0.2, 0 ] for i in range(len(edges))]  

               if category == 47 :
                  colors = [[0.4, 0.2, 0.2] for i in range(len(edges))] 
               
               if category == 48 :
                  colors = [[0.4, 0.2, 0.6] for i in range(len(edges))] 

               if category == 49 :
                  colors = [[0.4, 0.2, 0.8] for i in range(len(edges))] 

               if category == 50 :
                  colors = [[0.4, 0.4, 0.2] for i in range(len(edges))] 
                  
                  
               
               
               
               self.total_bboxes[self.count_bb].colors = o3d.utility.Vector3dVector(colors) 
               self.count_bb = self.count_bb + 1 
               
            
            for bbox in self.total_bboxes:
               self.__visualizer.update_geometry(bbox) 
               
            # remove existing text labels from the visualizer    
            for label in text_labels:
               self.__visualizer.remove_geometry(label) 
               
               
            
               
        
            
        self.__visualizer.poll_events()
        self.__visualizer.update_renderer()
        
        
        
       
            
















       
       