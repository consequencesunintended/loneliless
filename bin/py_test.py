import tensorflow as tf
import collections
import numpy as np
import cv2
import copy 

action_space_size = 2
lr = 0.01

source_name = 'source'
target_name = 'target'

def createNetowrk(name):
    with tf.variable_scope(name):

        input_images = tf.placeholder(tf.float32,shape=[None, 84,84,4])
        convo_1 = tf.layers.conv2d(inputs=input_images,filters=32,kernel_size=5,padding="same", activation=tf.nn.relu)
        convo_1_pooling = tf.layers.max_pooling2d(inputs=convo_1, pool_size=[2, 2], strides=2)
        convo_2 = tf.layers.conv2d(inputs=convo_1_pooling,filters=64,kernel_size=5,padding="same", activation=tf.nn.relu)
        convo_2_pooling = tf.layers.max_pooling2d(inputs=convo_2, pool_size=[2, 2], strides=2)
        convo_2_flat = tf.reshape(convo_2_pooling,[-1,21*21*64])
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

        return input_images, act, y_pred, y_true, train, enum_action

image_1, act, y_pred, y_true, train_step, enum_action = createNetowrk(source_name)
target_image_1, target_act, target_y_pred, target_y_true, target_train_step, target_enum_action = createNetowrk(target_name)

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
save_sync = 100
rewards_current_episodes = 0
rewards_all_episodes = []
num_episodes = 1000
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01
sum_losses = 0
loss_count = 0
replay_memory_max_size = 10000
replay_memory = collections.deque(maxlen=replay_memory_max_size)
total_reward = collections.deque(maxlen=100)
frame_buffer = collections.deque(maxlen=4)
sync_size = 1000
batch_size = 32
saver = tf.train.Saver()
Experience = collections.namedtuple("Experience", field_names=['state', 'action', 'reward', 'done', 'new_state'])
sess = tf.compat.v1.Session()
resized_screen = np.zeros((80,80))
sess.run(init)

def resize(frame):
    resized_screen = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(resized_screen, [80, 80, 1])
    return x_t.astype(np.uint8)

def buffer_frame(frame):
    frame = resize(frame)
    frame_buffer.append(frame)
    max_frame = np.dstack(frame_buffer)
    return max_frame

frame2 = np.zeros((80,80))

def get_action(frame):
    frame_buffer.append(frame)
    max_frame = np.dstack(frame_buffer)

    if len(frame_buffer) == 4:
        return max_frame[:,:,3]
    else:
        return max_frame[:,:,0]