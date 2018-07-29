#! /usr/bin/python3

import gym
import tensorflow as tf
import numpy as np

# Environment parameters
state_size = 4
action_size = 2

# Parameters
gamma = 0.99 # Factor to discount rewards by
hidden_size = 8 # Size of the hidden layer
learning_rate = 1e-2
num_episodes = 5000 # How long to train for
max_ep_length = 999 # Maximum length of a single episode
eps_between_updates = 5 # How many episodes we run before updating gradients

# Setup the environment
env = gym.make('CartPole-v0')

def discount_rewards(r):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0.0
    for i in reversed(range(0, len(r))):
        running_add *= gamma
        running_add += r[i]
        discounted_r[i] = running_add

    return discounted_r

# Define the agent
agent_state_in = tf.placeholder([None, state_size], dtype=tf.float32)
agent_W1 = tf.Variable(tf.truncated_normal([state_size, hidden_size], mean = 0.0, stddev = 1.0))
agent_b1 = tf.Variable(tf.zeros([hidden_size]))
agent_h = tf.nn.relu(agent_state_in * agent_W1 + agent_b1)
agent_W2 = tf.Variable(tf.truncated_normal([hidden_size, action_size], mean = 0.0, stddev = 1.0))
agent_b2 = tf.Variable(tf.zeros([action_size]))
agent_output = tf.nn.softmax(agent_h * agent_W2 + agent_h2)

agent_discounted_rewards = tf.placeholder(shape = [None], dtype = tf.float32)
agent_chosen_actions = tf.placeholder(shape = [None], dtype = tf.float32)

agent_indices = tf.range(0, tf.shape(agent_output)[0]) * tf.shape(agent_output)[1] + agent_chosen_actions
agent_responsible_outputs = tf.gather(tf.reshape(agent_output, [-1]), agent_indices)

agent_loss = -tf.reduce_mean(tf.log(agent_responsible_outputs)*agent_discounted_rewards)

agent_trainable_vars = tf.trainable_variables() # The variables we'll apply gradients to

agent_grad_holders = [] # Will be used to pass in the values for the gradients
for idx,var in enumerate(agent_trainable_vars):
    placeholder = tf.placeholder(tf.float32, name = idx + "_holder")
    agent_grad_holders.append(placeholder)

agent_gradients = tf.gradients(agent_loss, agent_trainable_vars)
agent_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
agent_update_batch = agent_optimizer.apply_gradients(zip(agent_gradients, agent_trainable_vars))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Set up variables to track performance
reward_history = []
length_history = []

ep = 0

grad_buffer = sess.run(tf.trainable_variables())
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0.0

while ep < num_episodes:
    state = env.reset()
    this_ep_history = []

    for j in range(max_ep_length):

        reward_this_ep = 0.0

        action_dist = sess.run(agent_output, feed_dict = {agent_state_in : state})
        chosen_action = np.random.choice(action_dist[0], p = action_dist[0])
        chosen_action = np.argmax(action_dist == chosen_action) # FIXME - I'm really not sure I get his line

        s1, r, d, _ = env.step(chosen_action)
        this_ep_history.append([s, a, r, s1])
        reward_this_ep += r
        state = s1

        if d == True: # FIXME - updates only happen when done, which we don't always reach
            # Update the network
            this_ep_history = np.array(this_ep_history)
            discounted_rewards = discount_rewards([:,2])
            feed_dict = {agent_discounted_rewards : this_ep_history[:,2],
                         agent_chosen_actions : this_ep_history[:,1],
                         agent_state_in : this_ep_history[:,0]}
            grads = sess.run(agent_gradients, feed_dict = feed_dict)
            for idx, grad in enumerate(grads):
                grad_buffer[idx] += grad

            if ep % eps_between_updates == 0 and ep != 0:
                feed_dict = dict(zip(agent_grad_holders, grad_buffer))
                sess.run(agent_update_batch, feed_dict = feed_dict)
                # Clear the gradient buffers
                for ix, grad in enumerate(grad_buffer):
                    grad_buffer[ix] = grad * 0.0

            reward_history.append(reward_this_ep)
            length_history.append(j)
            break

    if ep % 100 == 0:
        print("Trained on {} episodes".format(i))
        print("Mean reward: {}".format(np.mean(reward_history[-100:])))

    ep += 1
