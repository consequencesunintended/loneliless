
import collections
import numpy as np
import cv2
import copy 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K
import tensorflow.keras as keras

#print (tf.__version__)

action_space_size = 3
frames_to_store = 8

gamma = 0.99
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
exploration_rate = 1

train_episodes = 4000
episodes = 1
episode_t = 0
save_sync = 1000
total_training_episodes = 6000
rewards_current_episodes = 0.0
rewards_all_episodes = []
num_episodes = 1000
sync_size = 1000
batch_size = 32

replay_memory_max_size = 4000
replay_memory = collections.deque(maxlen=replay_memory_max_size)
total_reward = collections.deque(maxlen=100)

modelPath = ""
Experience = collections.namedtuple("Experience", field_names=['state', 'action', 'reward', 'done', 'new_state'])

frames_buffer = collections.deque(maxlen=frames_to_store)
empty_frame = np.zeros((80,80,frames_to_store))
frames_buffer.append(empty_frame)
current_frames = np.dstack(frames_buffer)   
prev_frames = np.dstack(frames_buffer)  


#--------------------------------------------------------------------------------

mean_square_error = tf.keras.losses.MeanSquaredError()
adam_optimizer = tf.keras.optimizers.Adam(1e-4)

#--------------------------------------------------------------------------------

def getNumFramesToStore():
    return frames_to_store

#--------------------------------------------------------------------------------

def createModel():
    inputs = Input(shape=(80,80,frames_to_store), dtype=tf.float32 )

    conv_1 = Conv2D(32, 5, padding="same", activation='relu')(inputs)
    conv_1 = MaxPooling2D(strides=2)(conv_1)

    conv_2 = Conv2D(64, 5, padding="same", activation='relu')(conv_1)
    conv_2 = MaxPooling2D(strides=2)(conv_2)
    conv_2_flat = Flatten()(conv_2)

    dense_1 = Dense(512, activation='relu')(conv_2_flat)
    y_pred = Dense(action_space_size)(dense_1)
    act = tf.math.argmax(y_pred, 1)

    enum_action = Input(shape=(2), dtype=tf.int32)
    gathered_layer = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0],x[1]))([y_pred, enum_action])

    model = Model(inputs=[inputs,enum_action], outputs=gathered_layer)

    action_model = Model(inputs=inputs, outputs=act)
    
    y_pred_model = Model(inputs=inputs, outputs=y_pred)

    return model, action_model, y_pred_model

source_model, source_action, source_y_pred = createModel()
target_model, target_action, target_y_pred = createModel()

source_vars = source_model.trainable_variables
target_vars = target_model.trainable_variables

#--------------------------------------------------------------------------------

def run_ops(target_vars, source_vars):
    ops = []
    for i in range(len(target_vars)):
        var = source_vars[i]
        var2 = target_vars[i]
        var2.assign(var)

#--------------------------------------------------------------------------------

def buffer_frame(frame):
    frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
    frames_buffer.append(frame)
    
    global current_frames
    global prev_frames
    prev_frames = current_frames
    current_frames = np.dstack(frames_buffer)      

#--------------------------------------------------------------------------------

def get_frame(index):    
    return current_frames[:,:,index]

#--------------------------------------------------------------------------------

def get_frames():   
    return current_frames

#--------------------------------------------------------------------------------

def get_action():

    frames = get_frames()
    action = [0]

    if (np.random.rand(1)) < exploration_rate:
        action[0] = np.random.randint(0, action_space_size)
    else:
        action = source_action(np.array([current_frames])).numpy()
    return action[0]

#--------------------------------------------------------------------------------

def model_loss(y_true, y_pred):
    return mean_square_error(y_true, y_pred)

#--------------------------------------------------------------------------------

@tf.function
def train_step(x_values, y_values):

    with tf.GradientTape() as gen_tape:
        y_pred = source_model(x_values, training=True)
        model_losses = model_loss(y_values,y_pred)

    gradients_of_generator = gen_tape.gradient(model_losses, source_model.trainable_variables)
    adam_optimizer.apply_gradients(zip(gradients_of_generator, source_model.trainable_variables))

#--------------------------------------------------------------------------------

def add_replay_memory(action, rew, done):
    global episode_t
    episode_t += 1
    done_value = 0.0 if done == True else 1.0
    exp = Experience(prev_frames, action, rew, done_value, current_frames)
    replay_memory.append(exp)

    can_sync = False

    global rewards_current_episodes
    rewards_current_episodes += rew
    
    if len(replay_memory) == replay_memory_max_size:

        indicies = np.random.choice(replay_memory_max_size,batch_size,replace=False)
        #indicies = np.random.randint(0,replay_memory_max_size,size=batch_size)
        prev_state_list, action_list, rew_list, done_list, next_state_list = zip(*[ replay_memory[x] for x in indicies])
        target_q = target_y_pred( np.array(next_state_list) ).numpy()
        target_q_val = np.max( target_q, axis=1 )
        action_array = []
        target_q_val = target_q_val * done_list
        q_true_values = rew_list + gamma * target_q_val
        for i in range(batch_size):
            action_array.append([i, action_list[i]])

        prev_state_list = np.array(prev_state_list)
        action_array = np.array(action_array)
        x_values = [prev_state_list,action_array]
        y_values = np.array(q_true_values)
        can_sync = True

        #using the fit function seems to be extremely slower than manually applying gradients
        #therefore using the train_step function
        train_step( x_values, y_values)
        #source_model.fit(x_values, y_values, verbose=0)    

    if (episode_t % sync_size == 0 and can_sync ):
        run_ops(target_vars, source_vars)


    if done:
        total_reward.append(rewards_current_episodes)
        global episodes
        print( "[ Episode: %d / %d ]: Mean Reward: %f " % (episodes, total_training_episodes, np.mean([total_reward[x] for x in range(len(total_reward))])) )
        rewards_current_episodes = 0        

        if episodes % save_sync == 0:
            source_model.save_weights( modelPath + '/source_model.h5')
            target_y_pred.save_weights( modelPath + '/target_model.h5')
            print("Model saved in path")

        if episodes == total_training_episodes:
            print("Model finished training")
            return True

        episodes += 1

        global exploration_rate
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episodes)
        #print("Exploration rate:{}".format( exploration_rate ) )

    return False

#--------------------------------------------------------------------------------

def setSavedModelPath(path):
    global modelPath
    modelPath = path

#--------------------------------------------------------------------------------

def restoreMode():
    source_model.load_weights( modelPath + '/source_model.h5')
    target_y_pred.load_weights( modelPath + '/target_model.h5')
    print("Model restored")

#--------------------------------------------------------------------------------

def get_trained_action():
    action = [0]
    action = source_action(np.array([current_frames])).numpy()
    return action[0]
