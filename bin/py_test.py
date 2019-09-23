import tensorflow as tf
import collections
import numpy as np
import cv2
import copy 

action_space_size = 3
lr = 0.01
frames_to_store = 8

source_name = 'source'
target_name = 'target'

def getFrameToStore():
    return frames_to_store

def createNetowrk(name):
    with tf.variable_scope(name):

        input_images = tf.placeholder(tf.float32,shape=[None, 80,80,frames_to_store])
        convo_1 = tf.layers.conv2d(inputs=input_images,filters=32,kernel_size=5,padding="same", activation=tf.nn.relu)
        convo_1_pooling = tf.layers.max_pooling2d(inputs=convo_1, pool_size=[2, 2], strides=2)
        convo_2 = tf.layers.conv2d(inputs=convo_1_pooling,filters=64,kernel_size=5,padding="same", activation=tf.nn.relu)
        convo_2_pooling = tf.layers.max_pooling2d(inputs=convo_2, pool_size=[2, 2], strides=2)
        convo_2_flat = tf.reshape(convo_2_pooling,[-1,20*20*64])
        hidden_1 = tf.layers.dense(convo_2_flat, 512, activation=tf.nn.relu)
        y_pred = tf.layers.dense(hidden_1, action_space_size)
        act = tf.argmax(y_pred, 1)
        enum_action = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        gathered_layer = tf.gather_nd(y_pred, indices=enum_action)
        y_true = tf.placeholder(shape=None, dtype = tf.float32)
        loss = tf.losses.mean_squared_error(labels=y_true,predictions=gathered_layer)
        varsList = tf.trainable_variables(scope=name)
        optimiser = tf.train.AdamOptimizer(1e-4)
        train = optimiser.minimize(loss, var_list=varsList)

        return input_images, act, y_pred, y_true, train, enum_action, loss

image_1, act, y_pred, y_true, train_step, enum_action, loss_source = createNetowrk(source_name)
target_image_1, target_act, target_y_pred, target_y_true, target_train_step, target_enum_action, loss_target = createNetowrk(target_name)

source_vars = tf.trainable_variables(scope='source')
target_vars = tf.trainable_variables(scope='target')

def get_ops(target_vars, source_vars):

    ops = []
    for i in range(len(target_vars)):
        var = source_vars[i]
        var2 = target_vars[i]
        ops.append(var2.assign(var))
    return ops


def run_ops(ops):
    for op in ops:
        sess.run(op)

Operations = get_ops(target_vars, source_vars)
init = tf.compat.v1.global_variables_initializer()

gamma = 0.99
train_episodes = 4000
episodes = 0
episode_t = 0
save_sync = 1000

rewards_all_episodes = []
num_episodes = 1000
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
sum_losses = 0
loss_count = 0
replay_memory_max_size = 4000
replay_memory = collections.deque(maxlen=replay_memory_max_size)
total_reward = collections.deque(maxlen=100)
frames_buffer = collections.deque(maxlen=frames_to_store)
sync_size = 1000
batch_size = 32
saver = tf.train.Saver()
Experience = collections.namedtuple("Experience", field_names=['state', 'action', 'reward', 'done', 'new_state'])
sess = tf.compat.v1.Session()
resized_screen = np.zeros((80,80))
sess.run(init)

empty_frame = np.zeros((80,80,frames_to_store))
frames_buffer.append(empty_frame)
current_frames = np.dstack(frames_buffer)   
prev_frames = np.dstack(frames_buffer)  


def resize(frame):
    resized_screen = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(resized_screen, [80, 80, 1])
    return x_t.astype(np.uint8)

def buffer_frame(frame):
    frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
    frames_buffer.append(frame)
    global current_frames
    global prev_frames
    prev_frames = current_frames
    current_frames = np.dstack(frames_buffer)   

def define_globals():
    global rewards_current_episodes
    rewards_current_episodes = 0.0

def get_frame(index):    
    return current_frames[:,:,index]

def get_frames():   
    return current_frames

old_exploration_rate = 0

def get_action():

    frames = get_frames()
    action = [0]

    if (np.random.rand(1)) < exploration_rate:
        action[0] = np.random.randint(0, action_space_size)
    else:
        action = sess.run(act, feed_dict={image_1: [current_frames]})

    global old_exploration_rate;
    if ( old_exploration_rate != exploration_rate ):
        print("exploration_rate=", exploration_rate)
        old_exploration_rate = exploration_rate
        print("replay memory size=", len(replay_memory))
        print("loss value=", loss_value)

    return action[0]


loss_value = 0.0

def add_replay_memory(action, rew, done):
    global episode_t
    episode_t += 1
    done_value = 0.0 if done == True else 1.0
    exp = Experience(prev_frames, action, rew, done_value, current_frames)
    replay_memory.append(exp)

    global rewards_current_episodes
    rewards_current_episodes += rew
    
    if len(replay_memory) == replay_memory_max_size:

        indicies = np.random.randint(0,replay_memory_max_size,size=batch_size)
        prev_state_list, action_list, rew_list, done_list, next_state_list = zip(*[ replay_memory[x] for x in indicies])
        target_q = sess.run(target_y_pred, feed_dict={target_image_1: np.asarray(next_state_list)})
        target_q_val = np.max( target_q, axis=1 )
        action_array = []
        target_q_val = target_q_val * done_list
        q_true_values = rew_list + gamma * target_q_val
        for i in range(batch_size):
            action_array.append([i, action_list[i]])
        sess.run(train_step, feed_dict={image_1: prev_state_list,enum_action: action_array,y_true: q_true_values})
        global loss_value
        loss_value = sess.run(loss_source, feed_dict={image_1: prev_state_list,enum_action: action_array,y_true: q_true_values})




    if episode_t % sync_size == 0:
        sess.run(Operations)
        print("swapped")


    if done:
        total_reward.append(rewards_current_episodes)
        global episodes
        print( "Episode:", episodes )
        print(np.mean([total_reward[x] for x in range(len(total_reward))]))
        rewards_current_episodes = 0
        episodes += 1

        if episodes % save_sync == 0:
            save_path = saver.save(sess, "/tmp2/Lone.ckpt")
            print("Model saved in path: %s" % save_path)

        global exploration_rate
        exploration_rate = min_exploration_rate + \
                            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episodes)