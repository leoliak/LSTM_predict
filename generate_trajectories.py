from __future__ import print_function
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import torch

def listerman(folder_path, type_of_file):
    lis = []
    c = []
    for file in os.listdir(folder_path):
        if file.endswith(type_of_file):
            lis.append(os.path.join(file))
    for fi in lis:
        filename_w_ext = os.path.basename(fi)
        filename, file_extension = os.path.splitext(filename_w_ext)
        c.append(folder_path + filename_w_ext)
       # d.append(os.path.abspath(filename_w_ext))
    return c, lis

def plotter(max_paths,sub,j,height,width):
    plt.figure(j)
    for i in range(1,max_paths+1):
        s1 = sub[sub['Path'] == i]
        plt.title("Person %i Trajectory"%j)
        plt.xlim(0,width)
        plt.ylim(0,height)
        plt.plot(s1.x, s1.y)
    plt.show()

def gener(data,height,width,X_big,Y_big,x_data,y_data,pick=0):
    x_data3 = []
    y_data3 = []
    per = max(data.Person)
    print('Persons in video: ',per)
    print('Height of video: ',height)
    print('Width f video: ',width)
    if pick!=0:
        plot=True
    else:
        plot=False
    for j in range(1,per+1):
        sub = data[data['Person'] == j]
        max_paths = max(sub.Path)
        for i in range(1,max_paths+1):
            sub2 = sub[sub['Path'] == i]   
            sub2.y = height - sub2.y
            X = (np.array(sub2.x))
            Y = (np.array(sub2.y))
            X_big.append(X)
            Y_big.append(Y)
        # Matlab starts (0,0) upper left on images, so i sub 720 height with y coords
        if plot==True:
            plotter(max_paths,sub2,j,height,width)
    x_max = max([len(a) for a in X_big])
    y_max = max([len(a) for a in Y_big])
    for i in range(0,len(X_big)):
        x_data3.append(np.pad(X_big[i], (0,x_max-len(X_big[i])),'constant'))
        y_data3.append(np.pad(Y_big[i], (0,y_max-len(Y_big[i])),'constant'))
#    x_data = np.asarray(x_data)
#    y_data = np.asarray(y_data)
    x_data = x_data3
    y_data = y_data3
    return X_big,Y_big,x_data,y_data

X_big = []
Y_big = []
x_data = []
y_data = []
r = 'D:\lstm_networks\datas\\'
path_data, onomata = listerman(r, ".csv")

arx = onomata.index('arxeio.csv')
onomata.pop(arx)
path_data.pop(arx)

data_info = pd.read_csv('arxeio.csv')
di = data_info.values
for id_read in range(0,len(di)):  
    pick = 0
    vid_name = di[id_read,0]
    vid_id = di[id_read,1]
    height = di[id_read,3]
    width = di[id_read,2] 
    print('---------- Video Name: ',vid_name,' ------------')
    t = 'gTruth_'+str(vid_id)+'.csv'
    vid = pd.read_csv(t)
    if vid_id == 10:
        pick = 0
    X_big,Y_big,x_data,y_data = gener(vid,height,width,X_big,Y_big,x_data,y_data,pick=pick)
#    X_big[id_read] = X_big2
#    Y_big[id_read] = Y_big2
#    x_data[id_read] = x_data2
#    y_data[id_read] = y_data2
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

data_x = (x_data).astype('float64')
data_y = (y_data).astype('float64')
torch.save(data_y, open('traindata_y.pt', 'wb'))
torch.save(data_x, open('traindata_x.pt', 'wb'))
