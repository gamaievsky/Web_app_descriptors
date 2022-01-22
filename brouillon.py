import pandas as pd
pd.options.mode.chained_assignment = None
#
dict = {'name':  ['a','a','a','b','b'],
          'col': [34.4, 8.4,2.1, 1.3, 7.8]
         }
df = pd.DataFrame(dict)
frames=[]
m = df[(df['name']=='a') & (df['name']=='a')]
print(m)


#
# import numpy as np
#
# onsets = []
# with open('assets/cadence.txt','r') as f:
#     print(type(f))
#     for line in f:
#         l = line.split()
#         onsets.append(float(l[0]))
#
# print(onsets[-1])
#
#
# print(4166.67 * 12)
