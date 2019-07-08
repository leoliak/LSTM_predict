from __future__ import print_function
import pandas as  pd
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

r = 'D:\lstm_networks\datas\\'
path_data, onomata = listerman(r, ".csv")

arx = onomata.index('arxeio.csv')
onomata.pop(arx)
path_data.pop(arx)

df = pd.DataFrame(columns=['Path','x','y'])
path_sum = 0
data_info = pd.read_csv('arxeio.csv')
di = data_info.values
ii = 1
for id_read in range(0,len(di)):  
    pick = 0
    vid_name = di[id_read,0]
    vid_id = di[id_read,1]
    height = di[id_read,3]
    width = di[id_read,2] 
    print('---------- Video Name: ',vid_name,' ------------')
    t = 'gTruth_'+str(vid_id)+'.csv'
    vid = pd.read_csv(t)
    if vid_id == 50:
        pick = 0
    data2 = vid.drop(['Frame','Time'], axis=1)
    per = max(data2.Person)
    for i in range(1,per+1):
        sub = data2[data2['Person'] == i]
        max_p = max(sub.Path)
        path_sum = path_sum + max_p
        for iii in range(1,max_p+1):
            sub2 = sub[sub['Path'] == iii]
            sub2 = sub2.drop(['Person'],axis=1)
            sub2.loc[sub2['Path'] == iii, 'Path'] = ii
            if ii==1:
                df = sub2
            else:
                df = df.append(sub2)
            ii = ii+1
df = df.reset_index()
df = df.drop(['index'],axis=1)
df.to_csv('datatrain.csv',index=False)
torch.save(df, open('traindata.pt', 'wb'))