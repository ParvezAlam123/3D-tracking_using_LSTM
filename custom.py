import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from Dataset import CustomData
from model import Network 
import matplotlib.pyplot as plt 
import numpy as np
import math 
import time
import open3d as o3d 
from vis_nuscene import NonBlockVisualizer_nuScene
from scipy.optimize import linear_sum_assignment 


velo_path = "/media/parvez_alam/Expansion1/TiHAN_LiDAR/Scene5/lidar"
label_path = "/media/parvez_alam/Expansion1/TiHAN_LiDAR/Scene5/label"





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


valid_ds = CustomData(velo_path, label_path)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False) 



loaded_checkpoint = torch.load("trained_model_2_kitti.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_2_kitti.pth")


model = Network() 
model.load_state_dict(torch.load("model_state_2_kitti.pth"))
model.to(device)
model.eval() 



optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)






def validation(model, valid_loader):
    num_car = 0
    num_motorcycle = 0 
    num_pedestrian = 0
    

    car_error = 0
    motorcycle_error = 0  
    pedestrian_error = 0 
     

    for n, data in enumerate(valid_loader):
           first_frame = data["bboxes_1"].float().to(device)
          
           second_frame = data["bboxes_2"].float().to(device)
           
           
           first_frame_data = [] 
           second_frame_data = [] 

           for i in range(len(first_frame)):
               x = first_frame[0][i][0]
               y = first_frame[0][i][1]
               z = first_frame[0][i][2]
               length = first_frame[0][i][3]
               width = first_frame[0][i][4]
               height = first_frame[0][i][5]
               yaw = first_frame[0][i][6]
               category = first_frame[0][i][7]
               id = first_frame[0][i][8]
    
               states = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
               delta_x , delta_y = model(states) 

               x = x + delta_x[0].item()
               y = y + delta_y[0].item() 

               first_frame_data.append([id, category, x, y])
           
           for i in range(len(second_frame)):
               x = second_frame[0][i][0]
               y = second_frame[0][i][1]
               z = second_frame[0][i][2]
               category = second_frame[0][i][7]
               id = second_frame[0][i][8]

               
               second_frame_data.append([id, category, x, y])

           for i in range(len(first_frame_data)):
               for j in range(len(second_frame_data)):
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 0 :
                       num_car = num_car + 1 
                       car_error = car_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                       print("hello car ")
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 1:
                       num_pedestrian = num_pedestrian + 1 
                       pedestrian_error = pedestrian_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                       print("hello pedestrian")

                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 2:
                       num_motorcycle = num_motorcycle + 1 
                       motorcycle_error = motorcycle_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                       print("hello motorcycle")
                   

    if num_car != 0 :
        car_rmse = math.sqrt(car_error / num_car)
        print("Car Error = ", car_rmse)
    
   

    if num_pedestrian != 0 :
        pedestrian_rmse = math.sqrt(pedestrian_error / num_pedestrian)
        print("Pedestrian error = ", pedestrian_rmse)

    if num_motorcycle != 0 :
        motorcycle_rmse = math.sqrt(motorcycle_error / num_motorcycle)
        print("Motorcycle error = ", motorcycle_rmse)



    

def validation_vis(model, valid_loader):
    obj = NonBlockVisualizer_nuScene() 
    
    available_id = {}
    past_center_x = [] 
    past_center_y = [] 
    id = 0
    for n, data in enumerate(valid_loader):
        if n == 0:
            bb_list_1 = data["bboxes_1"]
            points = data["points_1"][0].numpy() 

            bboxes = []
            for i in range(len(bb_list_1)):
               #id = bb_list_1[i][6][0]
               #category = bb_list_1[i][5][0] 
               
               x = bb_list_1[0][i][0]
               y = bb_list_1[0][i][1]
               z = bb_list_1[0][i][2]
               length = bb_list_1[0][i][3]
               width = bb_list_1[0][i][4] 
               height = bb_list_1[0][i][5] 
               yaw = bb_list_1[0][i][6] 

               sequence = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

               delta_x, delta_y = model(sequence)

               
               past_center_x.append(x + delta_x[0].item())
               past_center_y.append(y + delta_y[0].item())

               center = [x, y, z]

               bboxes.append([center, length, width, height, yaw, id])
               available_id[i] = id 
               id = id + 1 


            pcd = o3d.geometry.PointCloud() 
            pcd.points = o3d.utility.Vector3dVector(points)

            obj.update_renderer(pcd, bboxes) 

        else:
            bb_list_1 = data["bboxes_1"]
            points = data["points_1"][0].numpy() 


            current_center_x = [] 
            current_center_y = [] 

            for i in range(len(bb_list_1)):
               x = bb_list_1[0][i][0]
               y = bb_list_1[0][i][1]

               current_center_x.append(x)
               current_center_y.append(y)

            
            cost_matrix = np.zeros((len(past_center_x), len(current_center_x))) 

            for i in range(len(past_center_x)):
                for j in range(len(current_center_x)):
                    cost_matrix[i][j] = math.sqrt((current_center_x[j] - past_center_x[i])**2+(current_center_y[j] - past_center_y[i])**2) 
                
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  

            
            current_available_id = {} 
            available_id_current = [] 
            current_index = [] 
            for i in range(len(row_ind)):
                current_available_id[col_ind[i]] = available_id[row_ind[i]]
                available_id_current.append(available_id[row_ind[i]]) 
                current_index.append(col_ind[i])


           
            if available_id_current != [] :
                minimum_id = min(available_id_current) 
                if minimum_id == 0 :
                    id = max(available_id_current) + 1 
                else:
                    id = min(available_id_current) - 1
            
            else:
                id = 0 
            
            # meke past values empty 
            past_center_x = [] 
            past_center_y = [] 


            bboxes = []
            current_id_buffer = [] 
            for i in range(len(bb_list_1)):
               #id = bb_list_1[i][6][0]
               #category = bb_list_1[i][5][0] 
               
               x = bb_list_1[0][i][0]
               y = bb_list_1[0][i][1]
               z = bb_list_1[0][i][2]
               length = bb_list_1[0][i][3]
               width = bb_list_1[0][i][4] 
               height = bb_list_1[0][i][5] 
               yaw = bb_list_1[0][i][6] 

               

               sequence = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)

               delta_x, delta_y = model(sequence)

               
               past_center_x.append(x + delta_x[0].item())
               past_center_y.append(y + delta_y[0].item())

               in_flag = True 
               
               for j in range(len(current_index)):
                    if i == current_index[j]:
                        id_avl = current_available_id[i] 

                        center = [x, y, z]
                        bboxes.append([center, length, width, height, yaw, id_avl]) 
                        current_id_buffer.append(id_avl) 
                        
                        in_flag = False 
                        break 



               if in_flag == True :
                    center = [x, y, z] 
                    bboxes.append([center, length, width, height,yaw, id])

                    current_available_id[i] = id 
                    
                    current_id_buffer.append(id)
                    


                    if min(current_id_buffer) == 0 :
                        id = max(current_id_buffer) + 1
                    else:
                        id = min(current_id_buffer) - 1  
                    



             
            print("current available_id = ", current_available_id)
            available_id = current_available_id 

            pcd = o3d.geometry.PointCloud() 
            pcd.points = o3d.utility.Vector3dVector(points)

            obj.update_renderer(pcd, bboxes) 
            time.sleep(0.5)













validation_vis(model, valid_loader)


