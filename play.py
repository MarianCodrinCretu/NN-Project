
from network import Network
import math
import random
import numpy as np
import time
from network import Network
from ple import PLE
from ple.games.waterworld import WaterWorld
import pygame

game_rewards = {
    "tick": -0.1,  # each time the game steps forward in time the agent gets -0.01
    "positive": 100000.0,  # each time the agent collects a green circle
    "negative": -100000.0,  # each time the agent bumps into a red circle
}

game_global_size = 356
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
learning_rate_alfa = 0.005
nr_sensors_per_direction = 6
nr_directions = 23
dark_color = (128, 128, 128)
good_hit_color = (255, 255, 0)
bad_hit_color = (255, 255, 255)

first_layer_size = nr_directions  # nr directional sensors

max_creeps = 10


#file_name = input("Numele fisierului de importat reteaua")
network = Network(5, 5)
#network.load_model("model_16_12_20_55")
network.load_model("model_16_12_21_21")
game = WaterWorld(width=game_global_size, height=game_global_size, num_creeps=max_creeps)
p = PLE(game, fps=30, force_fps=True, display_screen=True,
        reward_values=game_rewards)


p.init()
actions = p.getActionSet()
actions = actions[:-1]
screen = game.screen

def make_NN_training_variable_from_game_state(state):
    result = [0 for _ in range(first_layer_size)]  # normalizing the speeds
    reward = [0.0 for _ in range(first_layer_size)]

    for direction in range(nr_directions):
        for sensor_no in range(nr_sensors_per_direction):
            x = state["player_x"] + game.AGENT_RADIUS
            y = state["player_y"] + game.AGENT_RADIUS
            x += game.AGENT_RADIUS * math.cos(direction) * (sensor_no + 1.5)
            y += game.AGENT_RADIUS * math.sin(direction) * (sensor_no + 1.5)
            for i in state["creep_pos"]["GOOD"]:
                if math.sqrt((i[0] - x) **
                             2 + (i[1] - y) ** 2) <= game.AGENT_RADIUS:
                    reward[direction] += (nr_sensors_per_direction + 1 - sensor_no)
                    result[direction] += 1
            for i in state["creep_pos"]["BAD"]:
                if math.sqrt((i[0] - x) **
                             2 + (i[1] - y) ** 2) <= game.AGENT_RADIUS:
                    reward[direction] -= (nr_sensors_per_direction + 1 - sensor_no)
                    result[direction] -= 1
            #if (sensor_no <= 1):
               # if too_close_to_screen_end([x, y]):
                  #  result[direction] -= 1
            # if (x * x + y * y)
    return np.array(result), reward


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

def get_action_NN(state):
    NN_state = make_NN_training_variable_from_game_state(state)[0]
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
        for direction in range(nr_directions):
            for sensor_no in range(nr_sensors_per_direction):
                x = state["player_x"] + game.AGENT_RADIUS
                y = state["player_y"] + game.AGENT_RADIUS
                x += game.AGENT_RADIUS * math.cos(direction) * (sensor_no + 1.5)
                y += game.AGENT_RADIUS * math.sin(direction) * (sensor_no + 1.5)
                color = dark_color
                for i in state["creep_pos"]["GOOD"]:
                    if math.sqrt((i[0] - x) **
                                 2 + (i[1] - y) ** 2) <= game.AGENT_RADIUS:
                        color = good_hit_color
                for i in state["creep_pos"]["BAD"]:
                    if math.sqrt((i[0] - x) **
                                 2 + (i[1] - y) ** 2) <= game.AGENT_RADIUS:
                        color = bad_hit_color
                pygame.draw.circle(screen, color, (int(x), int(y)), 4)  # Here <<<

        pygame.display.update()
        next_state = p.getGameState()
        state = next_state
    p.display_screen = False
    p.reset_game()



play_game(100000)


