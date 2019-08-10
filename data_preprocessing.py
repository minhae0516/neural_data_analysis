import pickle
from pandas import DataFrame
import numpy as np

"""
all data should be located in data folder.
all output data also will be located in data folder.

"""




POMDP = True # True if you use pomdp data. False if you use neural data (data structure is different)
ENCODING = True # build encoding data, the encoding data alwasy read neural data even though POMDP is True
DECODING = True
RECODING = True


if ENCODING:
    dataN_pkl_file = open('./data/w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl',
                          'rb')

    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()

    bbelief = dataN_pkl['beliefs']  # behavior belief, 200x500x2, and here 2 means belief for two boxes.
    bb = bbelief.reshape(-1, 2)  # 2 beliefs for two boxes
    neural_response = dataN_pkl['neural_response']  # neural response
    r = neural_response.reshape(-1, 300)  # 300 neurons

    # for file: w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl
    # build dataframe: first 500 belief 2 data is wrong(all zeros). remove it
    bb_df = DataFrame(bb[500:], columns=['behavior_belief1', 'behavior_belief2'])
    bb_df.to_csv(path_or_buf='./data/bb_df.csv', index=False)

    r_df = DataFrame(r[500:])  # no colurmn name
    r_df.to_csv(path_or_buf='./data/r_df.csv', index=False)

if POMDP:
    """
    dataN_pkl['observations'] (200x500x10): 200 is the number of sequence; 500 is the sequence length; 10 include: 
    [0] action, [1] reward, [2] location. [3] color 1, [4] color2, [5-9] distribution of actions.
    dataN_pkl['beliefs'] (200x500x22): The first two dimensions have the same meaning as above. For the third dimension, 
    the first two [0, 1] are the beliefs of the two boxes, 
    [2-11] is the belief distribution for the first box, 
    [12-21] is the belief distribution for the second box.
    """
    dataN_pkl_file = open('./data/pomdp_07172019(1604)_dataN_twoboxCol.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()



    if DECODING: # to build decoding data

        bbelief = dataN_pkl['beliefs']  # behavior belief, 200x500x2, and here 2 means belief for two boxes.
        bb = (bbelief[:, :, :2] + 0.5) / 10
        bb = bb.reshape(-1, 2)
        obs = dataN_pkl['observations']
        a = obs[:, :, 0].reshape(-1, 1) # action
        location = obs[:, :, 2].reshape(-1, 1) # location

        decoding_data = np.concatenate((bb, a, location), axis=1)
        decoding_data_df = DataFrame(decoding_data, columns=['box1 belief', 'box2 belief', 'action', 'location'])
        decoding_data_df.to_csv(path_or_buf='./data/pomdp_decoding_data.csv', index=False)



    if RECODING:

        """
        To build a data for recoding: 
        the recoding processs needs causal relationship between (bb_prev, bb_now)
        so I remove the edge data of every new game episode
        """
        bbelief = dataN_pkl['beliefs'] # behavior belief, 200x500x2, and here 2 means belief for two boxes.
        bb = (bbelief[:, :, :2] + 0.5) / 10
        observations = dataN_pkl['observations']
        obs = observations[:, :, :5]  # action, reward, location, color 1, color 2

        # because of this, start from 1 in the for loop below
        bb_prev = bb[0,:-1,:]
        bb_now = bb[0,1:,:]
        obs_prev = obs[0,:-1,:]
        obs_now = obs[0,1:,:]

        for i in range(1, bbelief.shape[0]): # start from 1 on purpose
            #print(i)
            bb_prev = np.concatenate((bb_prev, bb[i,:-1,:]), axis=0)
            bb_now = np.concatenate((bb_now, bb[i,1:,:]), axis=0)
            obs_prev = np.concatenate((obs_prev, obs[i, :-1, :]), axis=0)
            obs_now = np.concatenate((obs_now, obs[i, 1:, :]), axis=0)



        recoding_data_prev = np.concatenate((bb_prev, obs_prev),  axis=1)
        recoding_data_now = np.concatenate((bb_now, obs_now), axis=1)

        recoding_data_prev_df = DataFrame(recoding_data_prev,
                                columns=['behavior_belief1', 'behavior_belief2', 'action', 'reward', 'location', 'color 1',
                                         'color 2'])
        recoding_data_now_df = DataFrame(recoding_data_now,
                               columns=['behavior_belief1', 'behavior_belief2', 'action', 'reward', 'location', 'color 1',
                                        'color 2'])
        recoding_data_prev_df.to_csv(path_or_buf='./data/recoding_pomdp_all_prev_df.csv', index=False)
        recoding_data_now_df.to_csv(path_or_buf='./data/recoding_pomdp_all_now_df.csv', index=False)

else: #if you are neural data

    """
    Here 'dataN_pkl' is a then dictionary with keys: dict_keys(['observations', 'beliefs', 'trueStates', 'allData']).
    dataN_pkl['observations'] has shape 200x500x5, where 200 is the number of sequences, 500 is the length of one sequence, 5 means it contains action, reward, location, color of box1, color of box2.
    dataN_pkl['beliefs'] has shape 200x500x2, and here 2 means belief for two boxes.
    dataN_pkl['trueState'] is the binary true state information for each box, and dataN_pkl['allData'] is just a stack of the variables above.
    """

    dataN_pkl_file = open('./data/w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl', 'rb')
    dataN_pkl = pickle.load(dataN_pkl_file)
    dataN_pkl_file.close()


    if DECODING: # need belief, location
        bbelief = dataN_pkl['beliefs']  # behavior belief, 200x500x2, and here 2 means belief for two boxes.
        bb = bbelief.reshape(-1, 2)  # 2 beliefs for two boxes
        obs = dataN_pkl['observations']
        action = obs[:, :, 0]  # actions
        a = action.reshape(-1, 1)  # one action
        neural_response = dataN_pkl['neural_response']  # neural response
        r = neural_response.reshape(-1, 300)  # 300 neurons
        location = obs[:, :, 2].reshape(-1, 1)  # location

        decoding_data = np.concatenate((bb, a, location), axis=1)
        decoding_data_df = DataFrame(decoding_data, columns=['box1 belief', 'box2 belief', 'action', 'location'])
        decoding_data_df.to_csv(path_or_buf='./data/neural_decoding_data.csv', index=False)



    if RECODING:
        """
        To build a data for recoding
        """
        bbelief = dataN_pkl['beliefs']  # behavior belief, 200x500x2, and here 2 means belief for two boxes.
        obs = dataN_pkl['observations']  # action, reward, location, color 1, color 2


        # because of this, start from 2 in the for loop below (0 is skipped because the first 500 data are useless)
        bb_prev = bbelief[1, :-1, :]
        bb_now = bbelief[1, 1:, :]
        obs_prev = obs[1, :-1, :]
        obs_now = obs[1, 1:, :]


        for i in range(2, bbelief.shape[0]):  # start from 2 on purpose (the first 500 data are useless - skip 0)
            # print(i)
            bb_prev = np.concatenate((bb_prev, bbelief[i, :-1, :]), axis=0)
            bb_now = np.concatenate((bb_now, bbelief[i, 1:, :]), axis=0)
            obs_prev = np.concatenate((obs_prev, obs[i, :-1, :]), axis=0)
            obs_now = np.concatenate((obs_now, obs[i, 1:, :]), axis=0)

        recoding_data_prev = np.concatenate((bb_prev, obs_prev), axis=1)
        recoding_data_now = np.concatenate((bb_now, obs_now), axis=1)

        recoding_data_prev_df = DataFrame(recoding_data_prev,
                                          columns=['behavior_belief1', 'behavior_belief2', 'action', 'reward',
                                                   'location', 'color 1',
                                                   'color 2'])
        recoding_data_now_df = DataFrame(recoding_data_now,
                                         columns=['behavior_belief1', 'behavior_belief2', 'action', 'reward',
                                                  'location', 'color 1',
                                                  'color 2'])
        recoding_data_prev_df.to_csv(path_or_buf='./data/recoding_neural_all_prev_df.csv', index=False)
        recoding_data_now_df.to_csv(path_or_buf='./data/recoding_neural_all_now_df.csv', index=False)






    # bbelief = dataN_pkl['beliefs'] # behavior belief, 200x500x2, and here 2 means belief for two boxes.
    # bb = bbelief.reshape(-1,2) # 2 beliefs for two boxes
    # obs = dataN_pkl['observations']
    # action = obs[:,:,0] # actions
    # a = action.reshape(-1,1) # one action
    # neural_response = dataN_pkl['neural_response'] # neural response
    # r = neural_response.reshape(-1,300) #300 neurons
    #
    #
    # """
    # #build dataframe
    # bb_df = DataFrame(bb, columns=['behavior_belief1', 'behavior_belief2'])
    # bb_df.to_csv(path_or_buf='./data/bb_df.csv',index=False)
    #
    # a_df = DataFrame(a, columns=['action'])
    # a_df.to_csv(path_or_buf='./data/a_df.csv',index=False)
    #
    # r_df = DataFrame(r) # no colurmn name
    # r_df.to_csv(path_or_buf='./data/r_df.csv',index=False)
    # """
    #
    # # for file: w_neuralactivity_07262019(1441)_data07172019(1604)_agent08062019(2209)_twoboxCol.pkl
    # # build dataframe: first 500 belief 2 data is wrong(all zeros). remove it
    # bb_df = DataFrame(bb[500:], columns=['behavior_belief1', 'behavior_belief2'])
    # bb_df.to_csv(path_or_buf='./data/bb_df.csv',index=False)
    #
    # a_df = DataFrame(a[500:], columns=['action'])
    # a_df.to_csv(path_or_buf='./data/a_df.csv',index=False)
    #
    # r_df = DataFrame(r[500:]) # no colurmn name
    # r_df.to_csv(path_or_buf='./data/r_df.csv',index=False)
    #
    # """
    # # combine data
    # data = np.concatenate((bb, a, r), axis=1) # 303 element per row
    #
    # # make data name
    # column_names = ['behavior_belief1', 'behavior_belief2', 'action']
    # for num in range(r.shape[1]): #make 300 neuron name
    #     column_names.append('neural_response'+ str(num))
    #
    # #build dataframe
    # df = DataFrame(data, columns=column_names)
    # df.to_csv(path_or_buf='./data/pandas_data.csv',index=False)
    #
    # """
    #
    # trueStates = dataN_pkl['trueStates'] # true state (food in each box)
    # state = trueStates.reshape(-1,2)
    # """
    # data_comb = np.concatenate((state[500:], bb[500:], a[500:]), axis=1)
    # data_comb_df = DataFrame(data_comb, columns=['box1 state', 'box2 state', 'box1 belief', 'box2 belief', 'action'])
    # data_comb_df.to_csv(path_or_buf='./data/combined_data.csv', index=False)
    # """
    #
    # observations = dataN_pkl['observations']
    # obs = observations.reshape(-1,5)
    # all_data_comb = np.concatenate((state[500:], bb[500:], obs[500:]), axis=1)
    # all_data_comb_df = DataFrame(all_data_comb, columns=['box1 state','box2 state','box1 belief', 'box2 belief','action', 'reward', 'location', 'box1 color', 'box2 color'])
    # all_data_comb_df.to_csv(path_or_buf='./data/all_data.csv', index=False)
    #





print('data preprocessing is successfully done!')


