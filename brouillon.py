# import pandas as pd
# pd.options.mode.chained_assignment = None
# #
# dict = {'name':  ['a','a','a','b','b'],
#           'col': [34.4, 8.4,2.1, 1.3, 7.8]
#          }
# df = pd.DataFrame(dict)
# frames=[]
# max = df[df['name']=='a']['col'].max()
# min = df[df['name']=='a']['col'].min()
# inter = df[df['name']=='a']
# if (max-min)!= 0:
#     inter['col'] = (inter['col'] - min) / (max-min)
# frames.append(inter)
#
# max = df[df['name']=='b']['col'].max()
# min = df[df['name']=='b']['col'].min()
# inter = df[df['name']=='b']
# if (max-min)!= 0:
#     inter['col'] = (inter['col'] - min) / (max-min)
# frames.append(inter)
#
# df_norm = pd.concat(frames)
# print(df_norm)


import numpy as np

onsets = []
with open('assets/cadence.txt','r') as f:
    print(type(f))
    for line in f:
        l = line.split()
        onsets.append(float(l[0]))

print(onsets[-1])
