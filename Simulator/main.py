#TODO clean up code
#TODO average out scores over a number of attempts
#TODO new network viewer in tkinter

if __name__ == "__main__":
    #test settings
    network_count = 250 # the amount of networks in the population
    episode_length = 600 # the maximum length of an episode
    generation_length = 500 # the amount of generations to run
    n_steps = 25 # the number of steps each network is allowed to run for per episode step

    graph_frequency = 1 # how often a graph is made
    display_frequency = 1 # how often the best network of a generation is displayed

    node_count_punishment = 1 # how much networks are punished for having a high node count
                              # helps prevent useless nodes in a network by
                              # encouraging the removal of useless nodes
    connection_count_punishment = 0.4 # similar to above, but punishes useless connections

    node_add_chance = 250 # x/1000 chance that a node is added
    connection_add_chance = 300 # x/1000 chance that a connection is added
    node_remove_chance = 200 # x/1000 chance that a node is removed
    connection_remove_chance = 240 # x/1000 chance that a connection is removed

    max_mutations = 5 #the maximum number of mutations that happen per copied network per generation

    bias_alter_chance = 200 # x/1000 chance that a bias is altered
    weight_alter_chance = 500 # x/1000 chance that a weight is altered

    environment = 'LunarLander-v2' # the openai gym environment for the networks to train on

    #import openai-gym https://gym.openai.com/docs/
    import gymnasium as gym
    import random
    import math
    import time
    from network import Network
    import numpy as np
    from GraphUtil.line import Line
    from GraphUtil.graph import Graph

    import os
    os.system("") #have to do this bullshit for console colours i guess

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        BLINK = '\033[m'

    #start the test environment
    env = gym.make(environment)
    env.reset()

    #create the lines to store our statistics
    peak_fitness_line = Line("Peak Fitness", color="lightblue", linestyle="dashed")
    mean_fitness_line = Line("Mean Fitness", color="orange")
    quartile1_fitness_line = Line("1st Quartile Fitness", color="lightgreen")
    quartile2_fitness_line = Line("2nd Quartile Fitness", color="green")
    quartile3_fitness_line = Line("3rd Quartile Fitness", color="lightgreen")

    best_net_node_count_line = Line("Best Network Node Count", color="lightblue", linestyle="dashed")
    mean_node_count_line = Line("Mean Node count", color="orange")
    quartile1_node_count_line = Line("1st Quartile Node count", color="lightgreen")
    quartile2_node_count_line = Line("2nd Quartile Node count", color="green")
    quartile3_node_count_line = Line("3rd Quartile Node count", color="lightgreen")

    mean_time_taken_line = Line("Mean time taken", color="orange")
    quartile1_time_taken_line = Line("1st Quartile time taken", color="lightgreen")
    quartile2_time_taken_line = Line("2nd Quartile time taken", color="green")
    quartile3_time_taken_line = Line("3rd Quartile time taken", color="lightgreen")

    best_net_connections_line = Line("Best network connections", color="lightblue", linestyle="dashed")
    mean_connections_line = Line("Mean connections", color="orange")
    quartile1_connections_line = Line("1st Quartile connections", color="lightgreen")
    quartile2_connections_line = Line("2nd Quartile connections", color="green")
    quartile3_connections_line = Line("3rd Quartile connections", color="lightgreen")

    delta_time_line = Line("ΔTime for Generationn", color="orange")

    def sig_negative(x):
        return (2 / (1 + np.exp(-x))) - 1

    def sig(x):
        return 1 / (1 + np.exp(-x))

    #create the networks
    n_inputs = 8#env.observation_space
    n_outputs = 4#env.action_space
    print(env.action_space.sample())
    print(n_inputs, n_outputs)
    #best 134
    networks = [Network("Genomes/Default", n_inputs, n_outputs, sig_negative) for i in range(network_count)]

    for gen_i in range(generation_length):
        print(f"\n\n{bcolors.HEADER}================================\n\tGENERATION {gen_i}\n================================{bcolors.ENDC}\n")
        #store the time at the start of the generation
        time_start = time.time()

        #test the networks on the environment
        scores = []
        times_taken = []
        for net_i in range(len(networks)):
            #penalise the networks for size to discourage useless sections of brain
            total_reward = -networks[net_i].GetNodeCount() * node_count_punishment
            total_reward += -networks[net_i].GetConnectionCount() * connection_count_punishment

            observation, info = env.reset() #reset the environment
            total_time = 0

            for t in range(episode_length):
                act = np.argmax(networks[net_i].Step(observation, n_steps)) #get the network's response to observation
                observation, reward, terminated, truncated, info = env.step(act) #do an action
                total_reward += reward #store the reward for this iteration
                total_time += 1

                if terminated or truncated: #exit if we've finished the episode
                    break

            reward_col = bcolors.OKGREEN if total_reward > 0 else bcolors.FAIL
            print(f"Episode for network {bcolors.OKGREEN}{net_i}{bcolors.ENDC} in {bcolors.OKCYAN}{gen_i}{bcolors.ENDC} finished after {bcolors.WARNING}{t+1}{bcolors.ENDC} timesteps with {reward_col}{total_reward}{bcolors.ENDC} reward")

            times_taken.append(total_time)
            scores.append((net_i, total_reward))
            #no need to reset env, env is reset at the start of each environment step

        #store the time at the end of the generation
        time_end = time.time()

        #sort the networks based on their scores
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        #calculate our statistics
        peak_fitness = scores[0][1]
        mean_fitness = sum([score[1] for score in scores]) / len(scores)
        quartile1_fitness = (scores[math.floor(len(scores) / 4)][1] + scores[math.ceil(len(scores) / 4)][1]) / 2
        quartile2_fitness = (scores[2 * math.floor(len(scores) / 4)][1] + scores[2 * math.ceil(len(scores) / 4)][1]) / 2
        quartile3_fitness = (scores[3 * math.floor(len(scores) / 4)][1] + scores[3 * math.ceil(len(scores) / 4)][1]) / 2

        #get the node counts of all networks
        node_counts = sorted([net.GetNodeCount() for net in networks], reverse=True)
        #calculate node count statistics
        best_net_node_count = networks[scores[0][0]].GetNodeCount()
        mean_node_count = sum(node_counts) / len(node_counts)
        quartile1_node_count = (node_counts[math.floor(len(node_counts) / 4)] + node_counts[math.ceil(len(node_counts) / 4)]) / 2
        quartile2_node_count = (node_counts[2 * math.floor(len(node_counts) / 4)] + node_counts[2 * math.ceil(len(node_counts) / 4)]) / 2
        quartile3_node_count = (node_counts[3 * math.floor(len(node_counts) / 4)] + node_counts[3 * math.ceil(len(node_counts) / 4)]) / 2

        #calculate time statistics
        best_net_time_taken = times_taken[scores[0][0]]
        mean_time_taken = sum(times_taken) / len(times_taken)
        times_taken = sorted(times_taken, reverse=True)
        quartile1_time_taken = (times_taken[math.floor(len(times_taken) / 4)] + times_taken[math.ceil(len(times_taken) / 4)]) / 2
        quartile2_time_taken = (times_taken[2 * math.floor(len(times_taken) / 4)] + times_taken[2 * math.ceil(len(times_taken) / 4)]) / 2
        quartile3_time_taken = (times_taken[3 * math.floor(len(times_taken) / 4)] + times_taken[3 * math.ceil(len(times_taken) / 4)]) / 2

        best_net_connections = networks[scores[0][0]].GetConnectionCount()
        connections = sorted([net.GetConnectionCount() for net in networks], reverse=True)
        mean_connections = sum(connections) / len(connections)
        quartile1_connections = (connections[math.floor(len(connections) / 4)] + connections[math.ceil(len(connections) / 4)]) / 2
        quartile2_connections = (connections[2 * math.floor(len(connections) / 4)] + connections[2 * math.ceil(len(connections) / 4)]) / 2
        quartile3_connections = (connections[3 * math.floor(len(connections) / 4)] + connections[3 * math.ceil(len(connections) / 4)]) / 2

        #store our statistics
        peak_fitness_line.add_point(gen_i, peak_fitness)
        mean_fitness_line.add_point(gen_i, mean_fitness)
        quartile1_fitness_line.add_point(gen_i, quartile1_fitness)
        quartile2_fitness_line.add_point(gen_i, quartile2_fitness)
        quartile3_fitness_line.add_point(gen_i, quartile3_fitness)

        best_net_node_count_line.add_point(gen_i, best_net_node_count)
        mean_node_count_line.add_point(gen_i, mean_node_count)
        quartile1_node_count_line.add_point(gen_i, quartile1_node_count)
        quartile2_node_count_line.add_point(gen_i, quartile2_node_count)
        quartile3_node_count_line.add_point(gen_i, quartile3_node_count)

        mean_time_taken_line.add_point(gen_i, mean_time_taken)
        quartile1_time_taken_line.add_point(gen_i, quartile1_time_taken)
        quartile2_time_taken_line.add_point(gen_i, quartile2_time_taken)
        quartile3_time_taken_line.add_point(gen_i, quartile3_time_taken)

        best_net_connections_line.add_point(gen_i, best_net_connections)
        mean_connections_line.add_point(gen_i, mean_connections)
        quartile1_connections_line.add_point(gen_i, quartile1_connections)
        quartile2_connections_line.add_point(gen_i, quartile2_connections)
        quartile3_connections_line.add_point(gen_i, quartile3_connections)

        delta_time_line.add_point(gen_i, (time_end - time_start))

        #determine if we should make a graph
        if gen_i % graph_frequency == 0:
            #Fitness graph
            #create a graph
            g = Graph("Network Fitness Statistics", "Generation", "Fitness")

            #add the lines
            g.add_line(peak_fitness_line)
            g.add_line(mean_fitness_line)
            g.add_line(quartile1_fitness_line)
            g.add_line(quartile2_fitness_line)
            g.add_line(quartile3_fitness_line)

            #save the graph
            g.save_graph("graphs/Fitness.png")
            g.save_graph_data("graphs/Fitness.json")

            #Node count graph
            #create a graph
            g = Graph("Network Node Count Statistics", "Generation", "Node Count")

            #add the lines
            g.add_line(best_net_node_count_line)
            g.add_line(mean_node_count_line)
            g.add_line(quartile1_node_count_line)
            g.add_line(quartile2_node_count_line)
            g.add_line(quartile3_node_count_line)

            #save the graph
            g.save_graph("graphs/Node count.png")
            g.save_graph_data("graphs/Node count.json")

            #Time taken graph
            #create a graph
            g = Graph("Time taken Statistics", "Generation", "Timesteps")

            #add the lines
            g.add_line(mean_time_taken_line)
            g.add_line(quartile1_time_taken_line)
            g.add_line(quartile2_time_taken_line)
            g.add_line(quartile3_time_taken_line)

            #save the graph
            g.save_graph("graphs/Time taken.png")
            g.save_graph_data("graphs/Time taken.json")

            #Total time taken graph
            #create a graph
            g = Graph("ΔTime for Generation Statistics", "Generation", "ΔTime(seconds)")

            #add the line
            g.add_line(delta_time_line)

            #save the graph
            g.save_graph("graphs/Generation time taken.png")
            g.save_graph_data("graphs/Generation time taken.json")

            #Connections graph
            #create a graph
            g = Graph("Connections in network Statistics", "Generation", "N. Connections")

            #add the lines
            g.add_line(best_net_connections_line)
            g.add_line(mean_connections_line)
            g.add_line(quartile1_connections_line)
            g.add_line(quartile2_connections_line)
            g.add_line(quartile3_connections_line)

            #save the graph
            g.save_graph("graphs/Connections.png")
            g.save_graph_data("graphs/Connections.json")

        #save and report the best network
        networks[scores[0][0]].SaveNetwork(f"Networks/BestNetFromGen_{gen_i}")
        reward_col = bcolors.OKGREEN if scores[0][1] > 0 else bcolors.FAIL
        print(f"Generation finished with peak reward total {reward_col}{scores[0][1]}{bcolors.ENDC} from network {bcolors.OKGREEN}{scores[0][0]}{bcolors.ENDC}")

        #test and display the best network
        if gen_i % display_frequency == 0:
            env = gym.make(environment, render_mode="human")
            #penalise the networks for size to discourage useless sections of brain
            total_reward = -networks[net_i].GetNodeCount() * node_count_punishment
            total_reward += -networks[net_i].GetConnectionCount() * connection_count_punishment

            observation, info = env.reset() #reset the environment
            total_time = 0

            for t in range(episode_length):
                act = np.argmax(networks[net_i].Step(observation, n_steps)) #get the network's response to observation
                observation, reward, terminated, truncated, info = env.step(act) #do an action
                total_reward += reward #store the reward for this iteration
                total_time += 1

                if terminated or truncated: #exit if we've finished the episode
                    break

            #close the environment window for performance
            env.close()
            env = gym.make(environment)
            env.reset()



        #create a new network list with the successful networks
        good_networks = []
        for net_tuple in scores[:len(scores)//2]:
            good_networks.append(networks[net_tuple[0]])

        #randomly copy and mutate these networks to make up for the deficit
        new_networks = []
        for i in range(network_count - len(good_networks)):
            #copy the best good network that hasn't already been copied
            new_networks.append(good_networks[i].copy())

            #do a random amount of mutations
            for mut_i in range(int(random.randrange(1, max_mutations))):
                #decide if we should mutate the network's structure
                if random.randint(1, 1000) < connection_add_chance:
                    source_index = random.randrange(0, new_networks[-1].GetInputCount() + new_networks[-1].GetNodeCount())
                    target_index = random.randrange(0, new_networks[-1].GetNodeCount() + new_networks[-1].GetOutputCount())
                    new_networks[-1].AddConnection(source_index, target_index, (random.random()-0.5)*3.9)

                if random.randint(1, 1000) < node_add_chance:
                    node_index = random.randrange(0, new_networks[-1].GetNodeCount() + new_networks[-1].GetOutputCount())
                    if new_networks[-1].GetNodeConnectionCount(node_index) != 0:
                        connection_index = random.randrange(0, new_networks[-1].GetNodeConnectionCount(node_index))
                        new_networks[-1].AddNode(node_index, connection_index, (random.random()-0.5)*3.9)

                #decide if we should remove from the network's structure
                if random.randint(1, 1000) < connection_remove_chance:
                    node_index = random.randrange(0, new_networks[-1].GetNodeCount() + new_networks[-1].GetOutputCount())
                    if new_networks[-1].GetNodeConnectionCount(node_index) != 0:
                        connection_index = random.randrange(0, new_networks[-1].GetNodeConnectionCount(node_index))
                        new_networks[-1].RemoveConnection(node_index, connection_index)

                if random.randint(1, 1000) < node_remove_chance:
                    if new_networks[-1].GetNodeCount() != 0:
                        node_index = random.randrange(0, new_networks[-1].GetNodeCount())
                        new_networks[-1].RemoveNode(node_index)

                #decide if we should alter the network's properties
                if random.randint(1, 1000) < weight_alter_chance:
                    target_index = random.randrange(0, new_networks[-1].GetNodeCount() + new_networks[-1].GetOutputCount())
                    if new_networks[-1].GetNodeConnectionCount(target_index) != 0:
                        connection_index = random.randrange(0, new_networks[-1].GetNodeConnectionCount(target_index))
                        source_index = new_networks[-1].ConnectionIndexToSourceIndex(target_index, connection_index)
                        new_networks[-1].SetConnectionWeight(
                            target_index,
                            connection_index,
                            new_networks[-1].GetConnectionWeight(source_index, target_index) + (random.random() - 0.5) * 0.05
                        )

                if random.randint(1, 1000) < bias_alter_chance:
                    if new_networks[-1].GetNodeCount() != 0:
                        node_index = random.randrange(0, new_networks[-1].GetNodeCount())
                        new_networks[-1].SetNodeBias(
                            node_index,
                            new_networks[-1].GetNodeBias(node_index) + (random.random() - 0.5) * 0.05
                        )

        #store the next generation
        networks = good_networks + new_networks

    #test the networks on the environment
    scores = []
    env = gym.make(environment, render_mode="human")
    for net_i in range(len(networks)):
        total_reward = 0
        observation, reward, done, info = (0, 0, 0, 0), 0, False, {}
        for t in range(episode_length):
            observation, reward, done, info = env.step(int(round(networks[net_i].Step(observation, n_steps)[0]))) #do an action
            total_reward += reward #store the reward for this iteration

            if done: #exit if we've finished the episode
                print(f"Episode for network {bcolors.OKGREEN}{net_i}{bcolors.ENDC} finished after {bcolors.WARNING}{t+1}{bcolors.ENDC} timesteps with {bcolors.FAIL}{total_reward}{bcolors.ENDC} reward")
                break
        scores.append((net_i, total_reward))

        env.reset()

    env.close()
