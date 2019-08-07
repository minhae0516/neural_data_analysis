import pickle
from pandas import DataFrame
import numpy as np


dataN_pkl_file = open('./data/w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl', 'rb')
#dataN_pkl_file = open('./data/07262019(1441)_data07172019(1604)_agent07312019(1103)_twoboxCol.pkl', 'rb')

dataN_pkl = pickle.load(dataN_pkl_file)
dataN_pkl_file.close()


bbelief = dataN_pkl['beliefs'] # behavior belief, 200x500x2, and here 2 means belief for two boxes.
bb = bbelief.reshape(-1,2) # 2 beliefs for two boxes
obs = dataN_pkl['observations']
action = obs[:,:,0] # actions
a = action.reshape(-1,1) # one action
neural_response = dataN_pkl['neural_response'] # neural response
r = neural_response.reshape(-1,300) #300 neurons


"""
#build dataframe
bb_df = DataFrame(bb, columns=['behavior_belief1', 'behavior_belief2'])
bb_df.to_csv(path_or_buf='./data/bb_df.csv',index=False)

a_df = DataFrame(a, columns=['action'])
a_df.to_csv(path_or_buf='./data/a_df.csv',index=False)

r_df = DataFrame(r) # no colurmn name
r_df.to_csv(path_or_buf='./data/r_df.csv',index=False)
"""

# for file: w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl
# build dataframe: first 500 belief 2 data is wrong(all zeros). remove it
bb_df = DataFrame(bb[500:], columns=['behavior_belief1', 'behavior_belief2'])
bb_df.to_csv(path_or_buf='./data/bb_df.csv',index=False)

a_df = DataFrame(a[500:], columns=['action'])
a_df.to_csv(path_or_buf='./data/a_df.csv',index=False)

r_df = DataFrame(r[500:]) # no colurmn name
r_df.to_csv(path_or_buf='./data/r_df.csv',index=False)

"""
# combine data
data = np.concatenate((bb, a, r), axis=1) # 303 element per row

# make data name
column_names = ['behavior_belief1', 'behavior_belief2', 'action']
for num in range(r.shape[1]): #make 300 neuron name
    column_names.append('neural_response'+ str(num))

#build dataframe
df = DataFrame(data, columns=column_names)
df.to_csv(path_or_buf='./data/pandas_data.csv',index=False)

"""

trueStates = dataN_pkl['trueStates'] # true state (food in each box)
state = trueStates.reshape(-1,2)
"""
data_comb = np.concatenate((state[500:], bb[500:], a[500:]), axis=1)
data_comb_df = DataFrame(data_comb, columns=['box1 state', 'box2 state', 'box1 belief', 'box2 belief', 'action'])
data_comb_df.to_csv(path_or_buf='./data/combined_data.csv', index=False)
"""

observations = dataN_pkl['observations']
obs = observations.reshape(-1,5)
all_data_comb = np.concatenate((state[500:], bb[500:], obs[500:]), axis=1)
all_data_comb_df = DataFrame(all_data_comb, columns=['box1 state','box2 state','box1 belief', 'box2 belief','action', 'reward', 'location', 'box1 color', 'box2 color'])
all_data_comb_df.to_csv(path_or_buf='./data/all_data.csv', index=False)


print('data preprocessing is successfully done!')


"""
Here 'dataN_pkl' is a then dictionary with keys: dict_keys(['observations', 'beliefs', 'trueStates', 'allData']).
dataN_pkl['observations'] has shape 200x500x5, where 200 is the number of sequences, 500 is the length of one sequence, 5 means it contains action, reward, location, color of box1, color of box2.
dataN_pkl['beliefs'] has shape 200x500x2, and here 2 means belief for two boxes.
dataN_pkl['trueState'] is the binary true state information for each box, and dataN_pkl['allData'] is just a stack of the variables above.
"""