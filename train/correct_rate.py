#! -*- coding:utf-8 -*-

import numpy as np
import pickle
from IPython import embed

with open("result.dump") as f:
    result = pickle.load(f)

keys  = result.keys()
keys.sort()

for key in keys:
    result[key][1] 
    result[key][2] /= float(result[key][0])
    if len(result[key]) == 5:
        result[key][4] /= float(result[key][0])

one_list  = np.zeros((5))
and_list = np.zeros((5))
nm_or_list  = np.zeros((5))
m_or_list  = np.zeros((5))
        
for i, key in enumerate(keys):
    if len(result[key]) == 5:
        if result[key][1] == 10:
            m_or_list[4] += 1
        elif result[key][1] >= 7:
            m_or_list[3] += 1
        elif result[key][1] >= 4:
            m_or_list[2] += 1
        elif result[key][1] >= 1:
            m_or_list[1] += 1
        else:
            m_or_list[0] += 1

    elif "or" in key:
        if result[key][1] == 10:
            nm_or_list[4] += 1
        elif result[key][1] >= 7:
            nm_or_list[3] += 1
        elif result[key][1] >= 4:
            nm_or_list[2] += 1
        elif result[key][1] >= 1:
            nm_or_list[1] += 1
        else:
            nm_or_list[0] += 1

    elif "and" in key:
        if result[key][1] == 10:
            and_list[4] += 1
        elif result[key][1] >= 7:
            and_list[3] += 1
        elif result[key][1] >= 4:
            and_list[2] += 1
        elif result[key][1] >= 1:
            and_list[1] += 1
        else:
            and_list[0] += 1
                
    else:
        if result[key][1] == 10:
            one_list[4] += 1
        elif result[key][1] >= 7:
            one_list[3] += 1
        elif result[key][1] >= 4:
            one_list[2] += 1
        elif result[key][1] >= 1:
            one_list[1] += 1
        else:
            one_list[0] += 1

print "number of possible situations with a 1-objective instruction is {}".format(int(one_list.sum()))
print "number of possible situations with an AND-concatenated instruction is {}".format(int(and_list.sum()))
print "number of possible situations with an OR-concatenated instruction (unique) is {}".format(int(nm_or_list.sum()))
print "number of possible situations with an OR-concatenated instruction (ambiguous) is {}".format(int(m_or_list.sum()))

print "n_success   0     1-3    4-6    7-9    perfect(10)"
print "1-obj.   ", one_list
print "AND      ", and_list
print "OR (uni.)", nm_or_list
print "OR (amb.)", m_or_list
