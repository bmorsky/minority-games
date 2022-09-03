import itertools
import random
import math
import copy
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns


# convert a list of int into string
# e.g. [1,1,1] -> "111"
def num_list_to_str(num_list):
    return "".join(str(e) for e in num_list)


# Select best item in list. If has several best one:
# Choose from them randomly
def max_randomly(list_item, key_function):
    if len(list_item) == 1:
        return list_item[0]
    else:
        list_item.sort(key=key_function, reverse=True)
        item_iter = 0
        max_key = key_function(list_item[0])
        for item in list_item:
            if key_function(item) == max_key:
                item_iter += 1
        last_item_index = item_iter
        return list_item[random.randint(0, last_item_index - 1)]


def min_randomly(list_item, key_function):
    if len(list_item) == 1:
        return list_item[0]
    else:
        list_item.sort(key=key_function, reverse=False)
        item_iter = 0
        min_key = key_function(list_item[0])
        for item in list_item:
            if key_function(item) == min_key:
                item_iter += 1
        last_item_index = item_iter
        return list_item[random.randint(0, last_item_index - 1)]


class StrategyTable(object):
    def __init__(self, depth=3, weight=0):
        """
        :param depth: agent memory
        :return: None
        """

        combinations_string_list = [num_list_to_str(i) for i in itertools.product([0, 1], repeat=depth)]
        all_stategy = [dict(zip(combinations_string_list, x)) for x in itertools.product([0,1], repeat=2**depth)]
        # self.__strategy_table = {x: random.randint(0, 1) for x in combinations_string_list}
        self.__strategy_table = random.choice(all_stategy)
        self.__weight = weight

    @property
    def weight(self):
        return self.__weight

    @property
    def strategy_table(self):
        return self.__strategy_table

    # predict with a string format history input
    def predict(self, history):
        """
        :param history: past m winning groups
        :return: next decision
        """
        return self.__strategy_table[history]

    def update_weight(self, is_win):
        """
        :param is_win: boolean, Did the strategy win
        :return: adjust the weight
        """
        if is_win:
            self.__weight += 1
        else:
            self.__weight -= 1


class Agent(object):
    """
    base class for AgentWithStrategyTable
    Also, used in random simulation
    """

    def predict(self):
        return random.randint(0, 1)


class AgentWithStrategyTable(Agent):
    """
    composed with StrategyTable class
    inherited with Agent class
    """

    def __init__(self, depth=3, strategy_num=2):
        # last memory
        self.__history = None
        self.__depth = depth
        # init strategy_pool
        self.__strategy_pool = []
        self.__win_times = 0
        self.__last_choice = 0
        for x in range(strategy_num):
            strategy = StrategyTable(depth)
            self.__strategy_pool.append(strategy)

    @property
    def strategy_pool(self):
        return self.__strategy_pool
    @property
    def win_times(self):
        return self.__win_times

    # predict with memory, a list
    def predict(self, history):
        """
        :param history: last m winning group code, list of int
        :return:   next decision from a table which has the highest weight
        """
        history = num_list_to_str(history)
        if len(history) == self.__depth:
            self.__history = history
            strategy_choice = max_randomly(self.__strategy_pool, lambda x: x.weight)
            self.__last_choice = strategy_choice.predict(self.__history)
            return self.__last_choice
        else:
            raise Exception("agent memory input error")


    # result is winner room number
    # update weights of tables
    def get_winner(self, result):
        """
        :param result: the winning group of this round
        """
        for table in self.__strategy_pool:
            is_win = result == table.predict(self.__history)
            table.update_weight(is_win)
        if result == self.__last_choice:
            self.__win_times+=1


class MinorityGame(object):
    """
    a base class for minority Game
    """
    def __init__(self, agent_num, run_num):
        self.agent_num = agent_num
        self.run_num = run_num
        self.agent_pool = []
        self.win_history = np.zeros(run_num)

    @property
    def score_mean_std(self):
        """
        :return: the winner number mean and stdd
        """
        return self.win_history.mean(), self.win_history.std()


class MinorityGameWithStrategyTable(MinorityGame):
    """
    class used for run the minority game with StrategyTable
    """
    def __init__(self, agent_num, run_num, depth, *strategy_num):
        super(MinorityGameWithStrategyTable,self).__init__(agent_num, run_num)
        self.all_history = list()
        self.depth = depth
        self.strategy_num = strategy_num

        for x in range(depth):
            self.all_history.append(random.randint(0, 1))
        self.init_agents()

    def init_agents(self):
        """
        generate S tables for each agent
        if strategy_num has multiple variable, then the agent population will have
        different strategy number for each agent
        """
        for i in range(self.agent_num):
            if i < self.agent_num // len(self.strategy_num):
                self.agent_pool.append(AgentWithStrategyTable(self.depth, self.strategy_num[0]))
            else:
                self.agent_pool.append(AgentWithStrategyTable(self.depth, self.strategy_num[1]))

    # return all strategy list  
    def all_strategy(self):
        combinations_string_list = [num_list_to_str(i) for i in itertools.product([0, 1], repeat=3)]
        all_stategy = [dict(zip(combinations_string_list, x)) for x in itertools.product([0,1], repeat=8)]
        return all_stategy

    # imitation using fermi rule
    def imitate(self, immitation_rate, k, strategy_freq):
        n = len(self.agent_pool)
        for i in range(4*n):
        #for i in range(len(self.agent_pool)):
            if immitation_rate > random.uniform(0,1):
                # randomly select an agent
                agent_index = random.randint(0, len(self.agent_pool) - 1)
                # generate a random opponent
                opponent_index = random.choice([j for j in range(len(self.agent_pool)) if j is not agent_index])
                agent = max_randomly(self.agent_pool[agent_index].strategy_pool, lambda x: x.weight)
                opponent = max_randomly(self.agent_pool[opponent_index].strategy_pool, lambda x: x.weight)
                try:
                    fermi = 1/(1 + math.exp(k*(agent.weight - opponent.weight)))
                except:
                    fermi = float('inf')
                if fermi > random.uniform(0,1):
                    worst = min_randomly(self.agent_pool[agent_index].strategy_pool, lambda x: x.weight)

                    # update strategy frequency
                    all_stategy = self.all_strategy()
                    for i in range (len(all_stategy)):
                        if all_stategy[i] == opponent.strategy_table:
                            strategy_freq[i] += 1
                        if all_stategy[i] == worst.strategy_table:
                            strategy_freq[i] -= 1

                    # replace agent's worst with opponent's best
                    self.agent_pool[agent_index].strategy_pool.remove(worst)
                    self.agent_pool[agent_index].strategy_pool.append(opponent)

    # mutation
    def mutate(self, mutation_rate, strategy_freq):
        for i in range(len(self.agent_pool)):
            if mutation_rate > random.uniform(0,1):
                # randomly select an agent
                agent_index = random.randint(0, len(self.agent_pool) - 1)
                
                # randomly select an old strategy
                strategy_num = len(self.agent_pool[agent_index].strategy_pool)
                old_strat_index = random.randint(0, strategy_num - 1)
                old_strat = self.agent_pool[agent_index].strategy_pool[old_strat_index]

                # randomly generate a new strategy
                new_strat = StrategyTable(3,0)

                # update strategy frequency
                all_stategy = self.all_strategy()
                for i in range (len(all_stategy)):
                    if all_stategy[i] == new_strat.strategy_table:
                        strategy_freq[i] += 1
                    if all_stategy[i] == old_strat.strategy_table:
                        strategy_freq[i] -= 1
                
                # remove the old strategy of the agent and add the new strategy
                self.agent_pool[agent_index].strategy_pool.pop(old_strat_index)
                self.agent_pool[agent_index].strategy_pool.append(new_strat)

    # simulation
    def run_game(self):
        """
        run the minority game n times
        """
        mean_list = []
        stdd_list = []
        
        volatility=[]
        avolatility=[]
        zero_win = 0
        one_win = 0
        k = 10
        avg_win = [ [0]*11 for i in range(11)]
        initial_freq = [0]*256
        temp = 0
        shannon = []
        shanno = []

        # put initial frequency of strategies into strategy_freq
        for x in range(len(self.agent_pool)):
            for y in self.agent_pool[x].strategy_pool:
                all_stategy = self.all_strategy()
                for i in range (len(all_stategy)):
                    if all_stategy[i] == y.strategy_table:
                        initial_freq[i] += 1

        # save initial agents' info
        initial_agent_copy = copy.deepcopy(self.agent_pool)

        #simulation
        for j in np.arange(0.0,1.1,0.1):
            for h in np.arange(0.0,1.1,0.1):
                win_proportion = []
                num_of_one_list=[]
                vol=[]
                avol=[]
                # strategy_freq = initial_freq
                strategy_freq = initial_freq[:]
                self.agent_pool = copy.deepcopy(initial_agent_copy)
                # print("Before the game, strategy frequencies are:")
                # print(strategy_freq)
                for i in range(self.run_num):
                    self.imitate(j, k, strategy_freq)
                    self.mutate(h, strategy_freq)
                    num_of_one = 0
                    for agent in self.agent_pool:
                        predict_temp  = agent.predict(self.all_history[-self.depth:])
                        num_of_one += predict_temp
                    game_result = 1 if num_of_one < self.agent_num / 2 else 0
                    for agent in self.agent_pool:
                        agent.get_winner(game_result)
                    winner_num = num_of_one if game_result == 1 else self.agent_num - num_of_one
                    if game_result == 1:
                        one_win += 1
                    else:
                        zero_win += 1
                    self.win_history[i] = winner_num
                    self.all_history.append(game_result)
                    if (i+1)%10000 == 0:
                        mean_list.append(self.win_history[:i].mean())
                        stdd_list.append(self.win_history[:i].std())
                        print("%dth round"%i)
                    win_proportion.append(winner_num/len(self.agent_pool))
                    num_of_one_list.append(num_of_one)
                    strat_freq_sum = sum(strategy_freq)
                    shan=0
                    for i in range(len(strategy_freq)):
                        if strategy_freq[i] > 0:
                            shan -= (strategy_freq[i]/strat_freq_sum) * math.log(strategy_freq[i]/strat_freq_sum)
                    # print(shan)
                    shannon.append(shan)
                
                # print("After the game, strategy frequencies are:")
                # print(strategy_freq)
                c = sum(shannon)/len(shannon)
                shannon = []
                # sourceFile = open('reinitialze.txt', 'a')
                # print("social: ",j, "asocial: ", h, file = sourceFile)
                # print("average is: ", c, file = sourceFile)
                # print("counts", len(shannon), file = sourceFile)
                # print("--------------------------------", file = sourceFile)
                # sourceFile.close()
                shanno.append(c)
                xbar=sum(win_proportion)/len(win_proportion)
                abar=sum(num_of_one_list)/len(num_of_one_list)
                for i in range(self.run_num):
                    vol.append((win_proportion[i]-xbar)*(win_proportion[i]-xbar)/(((len(self.agent_pool)-1))*(len(self.agent_pool))))
                for i in range(self.run_num):
                    avol.append((num_of_one_list[i]-abar)*(num_of_one_list[i]-abar)/(((len(self.agent_pool)-1))*(len(self.agent_pool))))
                volatility.append(sum(vol))
                avolatility.append(sum(avol))
                avg_j_h = sum(win_proportion) / self.run_num
                j_index = int(j*10)
                h_index = int(h*10)
                avg_win[j_index][h_index] = avg_j_h
        #volatility=[volatility[x:x+11] for x in range(0, len(volatility), 11)]
        #avolatility=[avolatility[y:y+11] for y in range(0, len(avolatility), 11)]
        #shanno=[shanno[y:y+11] for y in range(0, len(shanno), 11)]
        print("The number of 0 win: " + str(zero_win))
        print("The number of 1 win: " + str(one_win))
        # print("After the game, strategy frequencies are:")
        # print(strategy_freq)
        return mean_list,stdd_list,avg_win,volatility,avolatility,shanno


    def winner_for_diff_group(self):
        mid = len(self.agent_pool)/len(self.strategy_num)
        first_part_score = 0
        second_part_score = 0
        index = 0
        for agent in self.agent_pool:
            if index<mid:
                first_part_score+=agent.win_times
            else:
                second_part_score+=agent.win_times
            index +=1

        return first_part_score,second_part_score


if __name__ == "__main__":
    mean_list = []
    stdd_list = []
    avg_win = []
    volatility = []
    avolatility = []
    shannon = []
    m = MinorityGameWithStrategyTable(101, 2, 3, 3) 
    # print(m.winner_for_diff_group())
    avevol=[]
    aveavol=[]
    aveshan=[]
    mean_list,stdd_list,avg_win,volatility,avolatility,shanno = m.run_game()
    avevol=volatility
    aveavol=avolatility
    aveshan=shanno
    #print(avevol)
    for i in range(2): #number of realization -1
        mean_list1 = []
        stdd_list1 = []
        avg_win1 = []
        volatility1 = []
        avolatility1 = []
        shannon1 = []
        mean_list1,stdd_list1,avg_win1,volatility1,avolatility1,shanno1 = m.run_game()
        avevol = np.add(avevol, volatility1)
        aveavol = np.add(aveavol, avolatility1)
        aveshan = np.add(avevol, shanno1)
    numreal = 3 #number of realization
    avevola = [x / numreal for x in avevol]
    avevolat = [avevola[x:x+11] for x in range(0, len(avevola), 11)] #final volatility after several realization
    aveavola = [x / numreal for x in aveavol]
    aveavolat = [aveavola[x:x+11] for x in range(0, len(aveavola), 11)] #final avolatility
    aveshann = [x / numreal for x in aveshan]
    aveshanno = [aveshann[x:x+11] for x in range(0, len(aveshann), 11)] #final shannon
    print(avevolat)
    print(aveavolat)
    print(aveshanno)


    #print(win_proportion)
    #print(avg_win)
    # print(avolatility)
    # print(volatility)
    

    # fig = plt.figure(figsize=(10, 4))
    # plt.plot(win_proportion)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)


    # fig.suptitle('Win Proportion')
    # plt.xlabel('Iteration number')
    # plt.ylabel('win proportion')
    # fig.savefig('change of win propor.jpg')
