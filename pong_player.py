import gym
import numpy as np

def downsample(image):
    return image[::2, ::2, :]

def remove_color(image):
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    processed_observation = input_observation[35:195] #crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)

    processed_observation[processed_observation != 0] = 1

    # convert 80 x 80 to 1600 x 1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)

    # store prev frame
    prev_processed_observation = processed_observation

    return input_observation, prev_processed_observation

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """compute the new hidden layer values and the new output layer values"""
    
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)

    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_val = np.random.uniform()
    if random_val < probability:
        return 2 # up
    else:
        return 3 # down

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()

    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)

    dC_dw1 = np.dot(delta_l2.T, observation_values)

    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer in weights.keys():
        g = g_dict[layer]
        expectation_g_squared[layer] = decay_rate * expectation_g_squared[layer] + (1 - decay_rate) * g**2
        weights[layer] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer] + epsilon))
        g_dict[layer] = np.zeros_like(weights[layer])

def discount_rewards(rewards, gamma):
    """ discounts rewards on previous actions based on how long ago they were taken"""

    discounted_rewards = np.zeros_like(rewards)

    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0
        
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount gradient with normalized rewards """

    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)

    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)

    return gradient_log_p * discounted_episode_rewards

def main():
    env = gym.make("Pong-v0")
    observation = env.reset()

    # parameters
    batch_size = 10
    gamma = 0.99
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }


    # for RMSProp algorithm
    expectation_g_squared = {}
    g_dict = {}
    for layer in weights.keys():
        expectation_g_squared[layer] = np.zeros_like(weights[layer])
        g_dict[layer] = np.zeros_like(weights[layer])

    episode_hidden_layer_values = []
    episode_observations = []
    episode_gradient_log_ps = []
    episode_rewards = []

    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
        
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)
        observation, reward, done, info = env.step(action) # carry out action

        reward_sum += reward
        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        if done:
            episode_number += 1

            # combine values for this episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            episode_gradient_log_ps_discounted = discount_with_rewards(
                episode_gradient_log_ps, 
                episode_rewards, 
                gamma)

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
            observation = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Resetting env. Episode reward total was %f. Running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            prev_processed_observations = None

main()

            




