import torch.nn as nn 
import torch 
from torch.utils.data import DataLoader
from Dataset import Data 
from model import Network 
import matplotlib.pyplot as plt 
import numpy as np
import math 
import time
import open3d as o3d 
from vis import NonBlockVisualizer
from scipy.optimize import linear_sum_assignment 


LABEL_DIR = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_label_2/training/label_02"
POINT_CLOUD_DIR ="/home/parvez_alam/Data/Kitti/Tracking/data_tracking_velodyne/training/velodyne"
CALIB_DIR = "/home/parvez_alam/Data/Kitti/Tracking/data_tracking_calib/training/calib"  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_ds = Data(velo_dir=POINT_CLOUD_DIR, label_dir=LABEL_DIR, calib_dir=CALIB_DIR)
train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=False) 

valid_ds = Data(velo_dir=POINT_CLOUD_DIR, label_dir=LABEL_DIR, calib_dir=CALIB_DIR, train=False)
valid_loader = DataLoader(dataset=valid_ds, batch_size=1, shuffle=False)


loaded_checkpoint = torch.load("trained_model_5.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_5.pth")


model = Network() 
model.load_state_dict(torch.load("model_state_5.pth"))
model.to(device)
model.eval() 



optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

training_loss = [] 


def train(model, train_loader, epochs):
    for i in range(epochs):
       running_loss = 0.0
       for n, data in enumerate(train_loader):
          frame_1 = data["first_frame"]
          frame_2 = data["second_frame"]
          frame_3 = data["third_frame"]
          frame_4 = data["fourth_frame"]
          frame_5 = data["fifth_frame"]
          frame_6 = data["sixth_frame"]
          frame_7 = data["seventh_frame"]
          frame_8 = data["eighth_frame"]
          frame_9 = data["ninth_frame"]
          frame_10 = data["tenth_frame"]
          frame_11 = data["eleventh_frame"]
          frame_12 = data["twelveth_frame"]
          frame_13 = data["thirteenth_frame"]
          frame_14 = data["fourteenth_frame"]
          frame_15 = data["fifteenth_frame"]
          frame_16 = data["sixteenth_frame"]
          frame_17 = data["seventeenth_frame"]
          frame_18 = data["eighteenth_frame"]
          frame_19 = data["ninteenth_frame"]
          frame_20 = data["twentyth_frame"]



          for k in range(len(frame_20)):
              id = frame_20[k][0][0]  

              
              # make the trajectory sequence
              x , y, z = frame_20[k][1][0]
              height = frame_20[k][2].item()
              width = frame_20[k][3].item()
              length = frame_20[k][4].item()
              yaw = frame_20[k][5].item()

              sequence = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)


              for l in range(len(frame_19)):
                  if frame_19[l][0][0] == id :
                      x, y, z = frame_19[l][1][0]
                      height = frame_19[l][2].item()
                      width = frame_19[l][3].item()
                      length = frame_19[l][4].item() 
                      yaw = frame_19[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x,y,z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)
                    

              for l in range(len(frame_18)):
                  if frame_18[l][0][0] == id :
                      x, y, z = frame_18[l][1][0]
                      height = frame_18[l][2].item()
                      width = frame_18[l][3].item()
                      length = frame_18[l][4].item()
                      yaw = frame_18[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)
              
              for l in range(len(frame_17)):
                  if frame_17[l][0][0] == id :
                      x, y, z = frame_17[l][1][0]
                      height = frame_17[l][2].item()
                      width = frame_17[l][3].item()
                      length = frame_17[l][4].item()
                      yaw = frame_17[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)
              
               
              for l in range(len(frame_16)):
                  if frame_16[l][0][0] == id :
                      x, y, z = frame_16[l][1][0]
                      height = frame_16[l][2].item()
                      width = frame_16[l][3].item()
                      length = frame_16[l][4].item()
                      yaw = frame_16[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)

              for l in range(len(frame_15)):
                  if frame_15[l][0][0] == id :
                      x, y, z = frame_15[l][1][0]
                      height = frame_15[l][2].item()
                      width = frame_15[l][3].item()
                      length = frame_15[l][4].item()
                      yaw = frame_15[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)

               
              for l in range(len(frame_14)):
                  if frame_14[l][0][0] == id :
                      x, y, z = frame_14[l][1][0]
                      height = frame_14[l][2].item()
                      width = frame_14[l][3].item()
                      length = frame_14[l][4].item()
                      yaw = frame_14[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)

              for l in range(len(frame_13)):
                  if frame_13[l][0][0] == id :
                      x, y, z = frame_13[l][1][0]
                      height = frame_13[l][2].item()
                      width = frame_13[l][3].item()
                      length = frame_13[l][4].item()
                      yaw = frame_13[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)
              
              for l in range(len(frame_12)):
                  if frame_12[l][0][0] == id :
                      x, y, z = frame_12[l][1][0]
                      height = frame_12[l][2].item()
                      width = frame_12[l][3].item()
                      length = frame_12[l][4].item()
                      yaw = frame_12[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0) 

              for l in range(len(frame_11)):
                  if frame_11[l][0][0] == id :
                      x, y, z = frame_11[l][1][0]
                      height = frame_11[l][2].item()
                      width = frame_11[l][3].item()
                      length = frame_11[l][4].item()
                      yaw = frame_11[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

                
              for l in range(len(frame_10)):
                  if frame_10[l][0][0] == id :
                      x, y, z = frame_10[l][1][0]
                      height = frame_10[l][2].item()
                      width = frame_10[l][3].item()
                      length = frame_10[l][4].item()
                      yaw = frame_10[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

              for l in range(len(frame_9)):
                  if frame_9[l][0][0] == id :
                      x, y, z = frame_9[l][1][0]
                      height = frame_9[l][2].item()
                      width = frame_9[l][3].item()
                      length = frame_9[l][4].item()
                      yaw = frame_9[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

              for l in range(len(frame_8)):
                  if frame_8[l][0][0] == id :
                      x, y, z = frame_8[l][1][0]
                      height = frame_8[l][2].item()
                      width = frame_8[l][3].item()
                      length = frame_8[l][4].item()
                      yaw = frame_8[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

              for l in range(len(frame_7)):
                  if frame_7[l][0][0] == id :
                      x, y, z = frame_7[l][1][0]
                      height = frame_7[l][2].item()
                      width = frame_7[l][3].item()
                      length = frame_7[l][4].item()
                      yaw = frame_7[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

               
              for l in range(len(frame_6)):
                  if frame_6[l][0][0] == id :
                      x, y, z = frame_6[l][1][0]
                      height = frame_6[l][2].item()
                      width = frame_6[l][3].item()
                      length = frame_6[l][4].item()
                      yaw = frame_6[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  


              for l in range(len(frame_5)):
                  if frame_5[l][0][0] == id :
                      x, y, z = frame_5[l][1][0]
                      height = frame_5[l][2].item()
                      width = frame_5[l][3].item()
                      length = frame_5[l][4].item()
                      yaw = frame_5[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0) 


              for l in range(len(frame_4)):
                  if frame_4[l][0][0] == id :
                      x, y, z = frame_4[l][1][0]
                      height = frame_4[l][2].item()
                      width = frame_4[l][3].item()
                      length = frame_4[l][4].item()
                      yaw = frame_4[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)   


              for l in range(len(frame_3)):
                  if frame_3[l][0][0] == id :
                      x, y, z = frame_3[l][1][0]
                      height = frame_3[l][2].item()
                      width = frame_3[l][3].item()
                      length = frame_3[l][4].item()
                      yaw = frame_3[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)   

               
              for l in range(len(frame_2)):
                  if frame_2[l][0][0] == id :
                      x, y, z = frame_2[l][1][0]
                      height = frame_2[l][2].item()
                      width = frame_2[l][3].item()
                      length = frame_2[l][4].item()
                      yaw = frame_2[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  


              for l in range(len(frame_1)):
                  if frame_1[l][0][0] == id :
                      x, y, z = frame_1[l][1][0]
                      height = frame_1[l][2].item()
                      width = frame_1[l][3].item()
                      length = frame_1[l][4].item()
                      yaw = frame_1[l][5][0].item()

                      sequence = torch.cat((sequence, torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0)), dim=0)  

              
              # chech the shape of seqeunce if it is less then 19 append with last sequence 
              time, dim = sequence.shape 
              if time < 19:
                rem = 19 - time 
                for rep in range(rem):
                    last_seq = sequence[-1, :].unsqueeze(dim=0)
                    sequence = torch.cat((sequence, last_seq), dim=0)
              
              # make batch size -> [B, Seq, dim]
              
              gt_delta_x = sequence[0][0] - sequence[1][0]
              gt_delta_y = sequence[0][1] - sequence[1][0]

              sequence = sequence[1:, :]

              sequence = sequence.unsqueeze(dim=0).to(device).float()

              delta_x, delta_y = model(sequence)

              
              loss = (delta_x - gt_delta_x)**2 + (delta_y - gt_delta_y)**2 

              # apply backprop 
              optimizer.zero_grad() 
              loss.backward() 
              optimizer.step() 

              running_loss = running_loss + loss.item() 

       training_loss.append(running_loss) 
       print("tranining_loss = {}, epoch = {} ".format(running_loss, i+1))

       checkpoint = {
            "epoch_number" : i+1,
            "model_state" : model.state_dict()
           }
       torch.save(checkpoint, "trained_model_5.pth") 



def validation_vis(model, valid_loader):
    obj = NonBlockVisualizer()
    id_buffer = [] 
    available_id = {} 
    id = 0 
    past_center_x = []
    past_center_y = [] 

    for n, data in enumerate(valid_loader):
        if n == 0:
            frame = data["first_frame"]
            points = data["points1"][0].numpy()
             

            bboxes = [] 
            for bb in  range(len(frame)):
                x, y, z = frame[bb][1][0]
                height = frame[bb][2].item()
                width = frame[bb][3].item() 
                length = frame[bb][4].item() 
                yaw = frame[bb][5].item() 
                box3d_corner = frame[bb][6][0].numpy()

                states = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
                delta_x , delta_y = model(states) 
                past_center_x.append(x + delta_x[0].item())
                past_center_y.append(y + delta_y[0].item()) 
                available_id[bb] = id 

                
                center = [x, y, z]
                bboxes.append([center, box3d_corner, id]) 
                
                id = id + 1 


            
            pcd = o3d.geometry.PointCloud() 
            pcd.points = o3d.utility.Vector3dVector(points)

            obj.update_renderer(pcd, bboxes)

        else:
            frame = data["first_frame"]

            points = data["points1"][0].numpy() 
            
            current_center_x = [] 
            current_center_y = [] 

            for bb in range(len(frame)):
                x,  y, z = frame[bb][1][0] 
                current_center_x.append(x)
                current_center_y.append(y)

            
            cost_matrix = np.zeros((len(past_center_x), len(current_center_x))) 

            for i in range(len(past_center_x)):
                for j in range(len(current_center_x)):
                    cost_matrix[i][j] = math.sqrt((current_center_x[j] - past_center_x[i])**2+(current_center_y[j] - past_center_y[i])**2) 
                
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  

            print("row_ind = ", row_ind, "col_ind = ", col_ind)
            current_available_id = {} 
            available_id_current = [] 
            current_index = [] 
            for i in range(len(row_ind)):
                current_available_id[col_ind[i]] = available_id[row_ind[i]]
                available_id_current.append(available_id[row_ind[i]]) 
                current_index.append(col_ind[i])


            print("current index = ", current_index)
            print("available id current = ", available_id_current)
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
            for bb in  range(len(frame)):
                x, y, z = frame[bb][1][0]
                height = frame[bb][2].item()
                width = frame[bb][3].item() 
                length = frame[bb][4].item() 
                yaw = frame[bb][5].item() 
                box3d_corner = frame[bb][6][0].numpy()

                states = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
                delta_x , delta_y = model(states) 
                past_center_x.append(x + delta_x[0].item())
                past_center_y.append(y + delta_y[0].item())  
                 
                in_flag = True 

                for i in range(len(current_index)):
                    if bb == current_index[i]:
                        id_avl = current_available_id[bb] 

                        center = [x, y, z]
                        bboxes.append([center, box3d_corner, id_avl]) 
                        current_id_buffer.append(id_avl)
                        in_flag = False 
                        break 

                if in_flag == True :
                    center = [x, y, z] 
                    bboxes.append([center, box3d_corner, id])

                    current_available_id[bb] = id 
                    
                    current_id_buffer.append(id)
                    
                    if min(current_id_buffer) == 0 :
                        id = max(current_id_buffer) + 1
                    else:
                        id = min(current_id_buffer) - 1 
                    

                    #id = id + 1 

            
            print("current available_id = ", current_available_id)
            available_id = current_available_id 

            pcd = o3d.geometry.PointCloud() 
            pcd.points = o3d.utility.Vector3dVector(points)

            obj.update_renderer(pcd, bboxes) 
            time.sleep(0.1)




def validation(model, valid_loader):
    num_car = 0
    num_van = 0 
    num_truck = 0
    num_pedestrian = 0
    num_cyclist = 0
    num_tram = 0
    num_person = 0 

    car_error = 0
    van_error = 0 
    truck_error = 0 
    pedestrian_error = 0 
    cyclist_error = 0 
    tram_error = 0 
    person_error = 0 

    for n, data in enumerate(valid_loader):
           first_frame = data["first_frame"]
           second_frame = data["second_frame"]
           first_frame_data = [] 
           second_frame_data = [] 

           for i in range(len(first_frame)):
               id = first_frame[i][0][0]
               category = first_frame[i][7][0]
               
               x, y, z = first_frame[i][1][0]
               height = first_frame[i][2].item()
               width = first_frame[i][3].item() 
               length = first_frame[i][4].item() 
               yaw = first_frame[i][5].item() 
    
               states = torch.tensor([x, y, z, height, width, length, yaw]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(device)
               delta_x , delta_y = model(states) 

               x = x + delta_x[0].item()
               y = y + delta_y[0].item() 

               first_frame_data.append([id, category, x, y])
           
           for i in range(len(second_frame)):
               id = second_frame[i][0][0]
               category = second_frame[i][7][0]
               
               x, y, z = second_frame[i][1][0] 
               second_frame_data.append([id, category, x, y])

           for i in range(len(first_frame_data)):
               for j in range(len(second_frame_data)):
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Car':
                       num_car = num_car + 1 
                       car_error = car_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                   
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Van':
                       num_van = num_van + 1 
                       van_error = van_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                   
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Truck':
                       num_truck = num_truck + 1 
                       truck_error = truck_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                   
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Pedestrian':
                       num_pedestrian = num_pedestrian + 1 
                       pedestrian_error = pedestrian_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                    
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Cyclist':
                       num_cyclist = num_cyclist + 1 
                       cyclist_error = cyclist_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 
                    
                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Tram':
                       num_tram = num_tram + 1 
                       tram_error = tram_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 

                   if first_frame_data[i][0] == second_frame_data[j][0] and first_frame_data[i][1] == 'Person':
                       num_person = num_person + 1 
                       person_error = person_error + (first_frame_data[i][2] - second_frame_data[j][2])**2 + (first_frame_data[i][3] - second_frame_data[j][3])**2 

    if num_car != 0 :
        car_rmse = math.sqrt(car_error / num_car)
        print("Car Error = ", car_rmse)
    
    if num_van != 0 :
        van_rmse = math.sqrt(van_error / num_van)
        print("Van error = ", van_rmse)

    if num_truck != 0 :
        truck_rmse = math.sqrt(truck_error / num_truck)
        print("Truck error = ", truck_rmse)

    if num_pedestrian != 0 :
        pedestrian_rmse = math.sqrt(pedestrian_error / num_pedestrian)
        print("Pedestrian error = ", pedestrian_rmse)


    if num_cyclist != 0 :
        cyclist_rmse = math.sqrt(cyclist_error / num_cyclist)
        print("cyclist error = ", cyclist_rmse)

    if num_tram != 0 :
        tram_rmse = math.sqrt(tram_error / num_tram)
        print("Tram error = ", tram_rmse)

    if num_person != 0 :
        person_rmse = math.sqrt(person_error / num_person)
        print("Person error = ", person_rmse)




validation_vis(model, valid_loader)

#validation(model, valid_loader)

                            
#train(model, train_loader, epochs=40)

# plot the loss curve 
#plt.plot(np.arange(40)+1, training_loss)
#plt.xlabel("number of epochs")
#plt.ylabel("training loss")
#plt.show() 





































            


            

               











   

   
   

   

   

   

   