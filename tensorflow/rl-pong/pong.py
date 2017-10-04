import numpy as np
import tensorflow as tf
import gym

OBSERVATION_DIM = 80*80
HIDDEN_DIM = 200
BATCH_SIZE = 10
NUM_BATCHES = 2000
GAMMA = 0.99 # for discounted reward
LEARNING_RATE = 3e-3
DECAY = 0.99 # for RMSProp

W1_SHAPE = (HIDDEN_DIM, OBSERVATION_DIM)
W2_SHAPE = (1, HIDDEN_DIM)

RENDER = False
RESTORE = True
SAVE_PATH = './model.ckpt'

# preprocessing taken from:
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel().reshape((OBSERVATION_DIM, 1))

# Build the graph
tf.reset_default_graph()
input_ = tf.placeholder(shape=(OBSERVATION_DIM, 1), dtype=tf.float32)
w1 = tf.Variable(tf.random_normal(shape=W1_SHAPE, mean=0.0, stddev=1/(tf.sqrt(tf.to_float(OBSERVATION_DIM)))))

hidden = tf.matmul(w1, input_)
hidden_out = tf.nn.relu(hidden)

w2 = tf.Variable(tf.random_normal(shape=W2_SHAPE, mean=0.0, stddev=1/(tf.sqrt(tf.to_float(HIDDEN_DIM)))))

activation = tf.matmul(w2, hidden_out)
# output is the probability of going up
output = tf.nn.sigmoid(activation)

# action_class is 1 or 0, for up or down
action_class = tf.placeholder(shape=(1,), dtype=tf.float32)
loss = - (action_class * tf.log(output) + (1 - action_class) * (1 - tf.log(output)))

grad_w1 = tf.placeholder(shape=W1_SHAPE, dtype=tf.float32)
grad_w2 = tf.placeholder(shape=W2_SHAPE, dtype=tf.float32)

optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY)

compute_gradients = optimizer.compute_gradients(loss, var_list=[w1, w2])
apply_gradients = optimizer.apply_gradients([(grad_w1, w1), (grad_w2, w2)])

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    if RESTORE:
        saver.restore(sess, SAVE_PATH)
        print(optimizer._learning_rate)
    else:
        sess.run(init)
    env = gym.make("Pong-v0")

    for i in range(NUM_BATCHES):
        batch_gradient_w1 = np.zeros(W1_SHAPE)
        batch_gradient_w2 = np.zeros(W2_SHAPE)

        for j in range(BATCH_SIZE):
            print('>>>>>>> {} / {} of batch {}'.format(j+1, BATCH_SIZE, i))
            state = env.reset()
            previous_x = None
            step_number = 0
            gradient_w1 = np.zeros(W1_SHAPE)
            gradient_w2 = np.zeros(W2_SHAPE)
            episode_reward = 0

            # The while loop for actions/steps
            while True:
                if RENDER:
                    env.render()

                current_x = prepro(state)
                observation = current_x - previous_x if previous_x is not None else np.zeros((OBSERVATION_DIM, 1))
                previous_x = current_x
                
                up_probability = sess.run(output, feed_dict={input_:observation})[0][0]

                # Open AI gym: 2: 'UP' and 3: 'DOWN'
                action = 2 if np.random.uniform() < up_probability else 3
                y = [1.0] if action == 2 else [0.0]

                state, reward, done, info = env.step(action)
                episode_reward += reward

                # calculate gradients after each action to be used later
                gradients = sess.run(compute_gradients, feed_dict={input_:observation, action_class:y})

                gradient_w1 += gradients[0][0]
                gradient_w2 += gradients[1][0]

                gradient_w1 *= GAMMA
                gradient_w2 *= GAMMA                

                if reward != 0:
                    gradient_w1 *= reward
                    gradient_w2 *= reward  

                    batch_gradient_w1 += gradient_w1
                    batch_gradient_w2 += gradient_w2               

                    gradient_w1 = np.zeros(W1_SHAPE)
                    gradient_w2 = np.zeros(W2_SHAPE)
                                
                if done:
                    print('\t\tepisode_reward: {}{}'.format(episode_reward, '' if episode_reward < 0 else ' <<---- WIN'))
                    break

        print('updating weights!!!')
        _ = sess.run(apply_gradients, feed_dict={grad_w1:batch_gradient_w1, grad_w2:batch_gradient_w2})

        print(np.mean(batch_gradient_w1), np.var(batch_gradient_w1))
        print(np.mean(batch_gradient_w2), np.var(batch_gradient_w2))

        save_path = saver.save(sess, SAVE_PATH)
        print('model saved: {}'.format(save_path))
