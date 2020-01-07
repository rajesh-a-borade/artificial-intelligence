# written by Rajesh Borade
 
import numpy as np
import time
 
#--------------Config parameters--------------------
REWARD_MATRIX = [[-1, 3, -1, 2, -1, -1, -1, -1, -1], 
[3, -1, 2, -1, 2, -1, -1, -1, -1], 
[-1, 2, -1, -1, -1, 2, -1, -1, -1], 
[2, -1, -1, -1, 2, -1, 3, -1, -1], 
[-1, 2, -1, 2, -1, 2, -1, 2, -1], 
[-1, -1, 2, -1, 2, -1, -1, -1, 2], 
[-1, -1, -1, 3, -1, -1, -1, 4, -1], 
[-1, -1, -1, -1, 2, -1, 4, -1, 10], 
[-1, -1, -1, -1, -1, 2, -1, 10, -1]]
 
MATRIX_SIZE = 9
FINAL_STATE = 8
INITIAL_STATE = 1
GAMMA = 0.8
R = np.matrix(REWARD_MATRIX)
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act
 
available_act = available_actions(INITIAL_STATE) 
 
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action
 
action = sample_next_action(available_act)
 
def update(current_state, action, GAMMA):    
    # print('current_state = ', current_state)
    # print('action = ', action)
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + GAMMA * max_value
 
update(INITIAL_STATE,action,GAMMA)
 
#-----------------------------------
# Training
#-----------------------------------
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,GAMMA)
     
print("Trained Q-matrix:")
print(Q/np.max(Q)*100)
# np.savetxt('./test.out', (Q/np.max(Q)*100), delimiter=',')
 
#-----------------------------------
# Testing
#----------------------------------- 
current_state = 0
steps = [current_state]
while current_state != FINAL_STATE:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index
print("Selected path:")
print(steps)

'''

Trained Q matrix:
[[  0.     62.96    0.     65.12    0.      0.      0.      0.      0.   ]
 [ 58.096   0.     60.96    0.     71.2     0.      0.      0.      0.   ]
 [  0.     60.96    0.      0.      0.     71.2     0.      0.      0.   ]
 [ 56.096   0.      0.      0.     71.2     0.     76.4     0.      0.   ]
 [  0.     60.96    0.     65.12    0.     71.2     0.     84.      0.   ]
 [  0.      0.     60.96    0.     71.2     0.      0.      0.     84.   ]
 [  0.      0.      0.     67.12    0.      0.      0.     88.      0.   ]
 [  0.      0.      0.      0.     71.2     0.     78.4     0.    100.   ]
 [  0.      0.      0.      0.      0.     71.2     0.    100.      0.   ]]

Selected path:
[0, 3, 6, 7, 8]

'''

