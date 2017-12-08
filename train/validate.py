#! -*- coding:utf-8 -*-

import os
import sys
import glob
import numpy as np
import pickle
from IPython import embed

rfs = glob.glob(sys.argv[1] + "/*")
tfs = glob.glob(sys.argv[2] + "/*")
rfs.sort()
tfs.sort()

pos = ["dd_", "du_", "ud_", "uu_"]
col = {"red":"R", "blue":"B", "green":"G"}
arm = {"left":12, "right":13}
var = {"d":-0.8, "u":0.8}
solve = {"up":"true", "down":"false"}
situation = {}

for i in range(len(rfs)):
    result = np.loadtxt(rfs[i])

    count = 50
    f = open(tfs[i], "r")
    line = f.readline()

    while line[0] == "#":
        info = line.split(" ")
        if len(info) < 3:
            break
        if count + int(info[3]) >= result.shape[0]:
            break

        # 現在の姿勢を決める
        posture = 0
        if result[count, 12] > 0.:
            posture += 2
        if result[count, 13] > 0.:
            posture += 1

        t_info = info[1].split("_")

        saddle = None
        
        # 3語指示の場合
        if int(info[3])-int(info[2]) == 12:
            if col[t_info[0]] == t_info[3][0]:
                obj_arm = "left"
                keep_arm = "right"
            else:
                obj_arm = "right"
                keep_arm = "left"
            if t_info[2] == solve[t_info[1]]:
                obj_var = 0.8
            else:
                obj_var = -0.8
            error1 = (result[count+int(info[3]), arm[obj_arm]] - obj_var)
            error2 = (result[count+int(info[3]), arm[keep_arm]] - var[pos[posture][arm[keep_arm]-12]])
            
        # and指示の場合
        elif "and" in info[1]:
            if t_info[4] == solve[t_info[3]]:
                obj_var = 0.8
            else:
                obj_var = -0.8
            error1 = (result[count+int(info[3]), 12] - obj_var)
            error2 = (result[count+int(info[3]), 13] - obj_var)

        # or指示かつduかudの場合
        elif ("or" in info[1]) and (pos[posture][0] != pos[posture][1]):
            error1 = (result[count+int(info[3]), 12] - var[pos[posture][0]])
            error2 = (result[count+int(info[3]), 13] - var[pos[posture][1]])
            
        # or指示かつddで動かない場合
        elif ("or" in info[1]) and (pos[posture] == "dd_") and (t_info[4] != solve[t_info[3]]):
            error1 = (result[count+int(info[3]), 12] - var[pos[posture][0]])
            error2 = (result[count+int(info[3]), 13] - var[pos[posture][1]])
                
        # or指示かつuuで動かない場合
        elif ("or" in info[1]) and (pos[posture] == "uu_") and (t_info[4] == solve[t_info[3]]):
            error1 = (result[count+int(info[3]), 12] - var[pos[posture][0]])
            error2 = (result[count+int(info[3]), 13] - var[pos[posture][1]])
            
        #or指示で動く場合
        else:
            error1 = (result[count+int(info[3]), 12] - 0.8)
            error2 = (result[count+int(info[3]), 13] - (-0.8))
            error3 = (result[count+int(info[3]), 12] - (-0.8))
            error4 = (result[count+int(info[3]), 13] - 0.8)
            if (error3**2 + error4**2) < (error1**2 + error2**2):
                error1 = error3
                error2 = error4
                saddle = 1
            else:
                saddle = 0

        if np.abs(error1) < 0.04 and np.abs(error2) < 0.04:
            success = 1
        else:
            success = 0
                
        sqrt_e = np.sqrt((error1**2+error2**2)/2)
        
        if situation.has_key(pos[posture]+info[1]):
            if situation[pos[posture]+info[1]][0] != 10:
                situation[pos[posture]+info[1]][0] += 1
                situation[pos[posture]+info[1]][1] += success
                situation[pos[posture]+info[1]][2] += sqrt_e
                if situation[pos[posture]+info[1]][3] < sqrt_e:
                    situation[pos[posture]+info[1]][3] = sqrt_e
                if saddle != None:
                    situation[pos[posture]+info[1]][4] += saddle
        else:
            # trial_num, success_num, total_error, worst_error, lr_rate(w.r.t or case)
            if saddle != None:
                situation[pos[posture]+info[1]] = [1, success, sqrt_e, sqrt_e, saddle]
            else:
                situation[pos[posture]+info[1]] = [1, success, sqrt_e, sqrt_e]

            
        count += int(info[3])
        line = f.readline()

with open('result.dump', 'w') as f:
    pickle.dump(situation, f)
