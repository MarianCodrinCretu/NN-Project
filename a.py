import math
import random
import numpy as np
import time
from network import Network
from ple import PLE
from ple.games.waterworld import WaterWorld

game_rewards = {
    "tick": -0.1,  # each time the game steps forward in time the agent gets -0.01
    "positive": 100000.0,  # each time the agent collects a green circle
    "negative": -100000.0,  # each time the agent bumps into a red circle
}  # THIS IS NOT EVEN USED ???

# make a PLE instance.
# use lower fps so we can see whats happening a little easier
max_creeps = 10
game_global_size = 356
game = WaterWorld(width=game_global_size, height=game_global_size, num_creeps=max_creeps)
p = PLE(game, fps=30, force_fps=True, display_screen=False,
        reward_values=game_rewards)

nr_episodes = 20
nr_steps_per_episode = 1000
min_nr_training_states = 2000
nr_states_for_training = 200
current_NN_states = list()  # touple of elements :
# state, action(0->4), next_state, reward_of_next_state
max_speed = game_global_size * 8.3  # parameters to normalize Imput in NN
max_distance = (game_global_size * game_global_size * 2) ** 0.5
action_to_number = dict()
gama = 0.9
epsilon = 1
learning_rate_alfa = 0.01

first_layer_size = 2 + 2 * max_creeps
network = Network(first_layer_size, 5, learning_rate_alfa)


# STATE DEFINITION : vx, vy, distances to the num_creeps green (or 0), distance to red (or 0)

p.init()
actions = p.getActionSet()
actions = actions[:-1]
# removing standing still action
for i in range(len(actions)):
    action_to_number[actions[i]] = i

start_time = time.time()


def make_NN_training_variable_from_game_state(state):
    result = []  # normalizing the speeds
    result.append((state["player_velocity_x"] + max_speed) / (2 * max_speed * 0.7))
    result.append((state["player_velocity_y"] + max_speed) / (2 * max_speed * 0.7))

    for it in p.getGameState()["creep_dist"]["GOOD"]:
        result.append(1 / it)
    while len(result) < 2 + max_creeps:
        result.append(0)
    for it in p.getGameState()["creep_dist"]["BAD"]:
        result.append(it / max_distance / 0.95)
    while len(result) < 2 + 2 * max_creeps:
        result.append(0)
    return np.array(result)


def validate_action(state, action):  # HERE HERE
    if (action == game.actions["up"]):
        if state["player_y"] - game.AGENT_RADIUS <= game.AGENT_RADIUS * 0.3:
            return False
        return True
    if (action == game.actions["left"]):
        if state["player_x"] - game.AGENT_RADIUS <= game.AGENT_RADIUS * 0.3:
            return False
        return True
    if (action == game.actions["right"]):
        if state["player_x"] + game.AGENT_RADIUS >= game_global_size - game.AGENT_RADIUS * 2:
            return False
        return True
    if (action == game.actions["down"]):
        if state["player_y"] + game.AGENT_RADIUS >= game_global_size - game.AGENT_RADIUS * 2:
            return False
        return True
    return False


def get_next_random_action(state):
    ok = False
    action = None
    while not ok:
        action = actions[np.random.randint(0, len(actions))]
        ok = validate_action(state, action)
    return action


def get_next_epsilon(x):  # CHANGE THIS AFTER YOU FINISH
    return max(0, x - 1 / (nr_steps_per_episode * 3))


def get_action_NN(state):
    NN_state = make_NN_training_variable_from_game_state(state)
    Q_for_state = network.result(NN_state.reshape(1, first_layer_size))
    for it in Q_for_state[0]:
        if math.isnan(it):
            print("NAN ERROR, NAN ERROR!!!")
            exit(0)
    states = list(range(len(actions)))
    states.sort(key=lambda i: Q_for_state[0][i], reverse=True)
    for i in states:
        if validate_action(state, actions[i]):
            return actions[i]
    return actions[states[1]]  # it should never arrive here


def play_game(seconds):
    print("We are playing for " + str(seconds) + "seconds")
    p.display_screen = True
    p.reset_game()
    p.reset_game()
    state = p.getGameState()
    start_time = time.time()
    for i in range(1, 100000):
        if (time.time() - start_time >= seconds):
            break
        if p.game_over() or i % 1000 == 0:
            print("GAME OVER")
            p.reset_game()  # just in case the game ends(improbable?? Or not apparently)
        action = get_action_NN(state)
        reward = p.act(action)
        next_state = p.getGameState()
        state = next_state
    p.display_screen = False
    p.reset_game()

# running the game
p.act(-1)
# this line fixes a weird error(first reward is 1.99)(WHY?? no idea)
state = p.getGameState()
reward = 0.0
oldest_state = 0
for episode in range(nr_episodes):
    p.reset_game()
    if (episode == nr_episodes / 2):
        play_game(10)
    for step in range(nr_steps_per_episode):
        epsilon = get_next_epsilon(epsilon)
        if p.game_over():
            print("GAME OVER")
            p.reset_game()  # just in case the game ends(improbable?? Or not apparently)
        if random.random() < epsilon:
            action = get_next_random_action(state)
        else:
            action = get_action_NN(state)
        reward = p.act(action)

        if (reward > 0):
            reward = 1000
        elif (reward < -0.2):
            reward = -1000
        else:
            reward = -0.001
        next_state = p.getGameState()
        if (len(current_NN_states) < min_nr_training_states):
            current_NN_states.append((make_NN_training_variable_from_game_state(state),
                                      action,
                                      make_NN_training_variable_from_game_state(next_state),
                                      reward))
        else:
            current_NN_states[oldest_state] = (make_NN_training_variable_from_game_state(state),
                                               action,
                                               make_NN_training_variable_from_game_state(next_state),
                                               reward)
            oldest_state += 1
            if oldest_state >= min_nr_training_states:
                oldest_state = 0
        if (len(current_NN_states) >= min_nr_training_states and step % 20 == 0):
            training_states = list()
            target_states = list()

            for state_nr in random.sample(range(0, min_nr_training_states), nr_states_for_training):
                training_states.append(current_NN_states[state_nr][0])
                Qs_for_state_prime = network.result(current_NN_states[state_nr][0].reshape(1, first_layer_size))
                target_states.append(Qs_for_state_prime.reshape(5))
                action_no = action_to_number[current_NN_states[state_nr][1]]
                target_states[len(target_states) - 1][action_no] = 0
                Qs_for_state_prime = network.result(current_NN_states[state_nr][2].reshape(1, first_layer_size))
                Q_for_state_prime_max = max(Qs_for_state_prime[0])
                target_states[len(target_states) - 1][action_no] = gama * Q_for_state_prime_max + \
                                                                   current_NN_states[state_nr][3]

            network.train(np.array(training_states), np.array(target_states))
        state = next_state
    print(str(episode) + " " + str(time.time() - start_time) + " seconds")

print("WE REACHED THE END, YEEEEY, now the network test is starting")
print("--- %s seconds --- FOR TRAINING" % (time.time() - start_time))
# network.save_model()

play_game(100)
