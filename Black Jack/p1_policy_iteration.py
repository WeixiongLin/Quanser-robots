from p1_util import *
import numpy as np
import random, copy
from enum import Enum


class Color(Enum):
    RED = 0
    BLACK = 1


class Card(object):
    def __init__(self, color=None):
        self.value = self._get_random_value()
        if color == Color.BLACK or color == Color.RED:
            self.color = color
        else:
            self.color = self._get_random_color()

    def _get_random_value(self):
        """Generates integers between 1 and 10."""
        return random.randint(1, 10)

    def _get_random_color(self):
        """Generates random colors.

        Color.RED with 1/3 and Color.BLACK with 2/3 probability.
        """
        random_number = random.random()
        if random_number <= 1 / 3.0:
            return Color.RED
        else:
            return Color.BLACK


class Deck(object):
    def take_card(self, color=None):
        return Card(color)


class State(object):
    def __init__(self, dealer_sum=0, agent_sum=0, is_terminal=False):
        self.dealer_sum = dealer_sum
        self.agent_sum = agent_sum
        self.is_terminal = is_terminal


class Action(Enum):
    STICK = 0
    HIT = 1


class Player(object):
    """This is a general class for a player of Easy21."""

    def policy(self, s):
        """Given the current state choose the next action."""
        raise NotImplemented()


class Dealer(Player):
    def policy(self, s):
        """Dealers policy as described in the assigment."""
        if s.dealer_sum >= 16:
            return Action.STICK
        else:
            return Action.HIT


class Environment(object):
    """An environment for the game Easy21."""

    def __init__(self):
        # the environment includes the dealer and the deck
        self.dealer = Dealer()
        self.deck = Deck()

        self.agent_max_value = 21  # max value an agent can get during the game
        self.dealer_max_value = 10  # max value the dealer can get when taking the first card
        self.actions_count = 2  # number of possible actions

    def check_bust(self, player_sum):
        return player_sum < 1 or player_sum > 21

    def generate_reward_bust(self, s):
        if s.agent_sum > s.dealer_sum:
            return 1
        elif s.agent_sum == s.dealer_sum:
            return 0
        else:
            return -1

    def take_card(self, card_color=None):
        """Returns a card from the deck."""
        Card = self.deck.take_card(card_color)
        return Card.value if Card.color == Color.BLACK else Card.value * -1

    def dealer_turn(self, s):
        """A full implementation of the dealer turn.

           The dealer turn starts when the agent sticks and
           ends when the dealer action is busted or action = sticks.
        """
        action = None
        while not s.is_terminal and action != Action.STICK:
            action = self.dealer.policy(s)
            if action == Action.HIT:
                s.dealer_sum += self.take_card()
            s.is_terminal = self.check_bust(s.dealer_sum)
        return s

    def initial_state(self):
        """In the beginning both the agent and the dealer take a card."""
        return State(self.take_card(Color.BLACK), self.take_card(Color.BLACK))

    def endProb(self,simTimes=int(1e5),verbose=False):
        '''calculate the prob of each terminal situation, p[i][j]means prob of starting with value i, ending with value j,
        p_bust[i] is bust prob starting with value i'''
        self.p=np.zeros((self.dealer_max_value+1,self.agent_max_value+1))
        self.p_bust=np.zeros((self.dealer_max_value+1))
        sim_times=simTimes
        for start_v in range(1,self.dealer_max_value+1):
            if verbose:
                print(f'calculating end prob of start_v={start_v}')
            count=np.zeros(self.agent_max_value+1)
            bust_t=0
            for t in range(sim_times):
                state=State(dealer_sum=start_v)
                end_state=self.dealer_turn(state)
                if self.check_bust(end_state.dealer_sum):
                    bust_t+=1
                else:
                    count[end_state.dealer_sum]+=1
            for end_v in range(1,self.agent_max_value+1):
                self.p[start_v][end_v]=count[end_v]/sim_times
            self.p_bust[start_v]=bust_t/sim_times
        if verbose:
            print(self.p,self.p_bust)

    def getNewStateAndProbReward(self,state,action):
        '''return a list of (newState,prob,reward)'''
        ans=[]
        if action==Action.STICK.value:
            for end_v in range(self.agent_max_value+1):
                prob=self.p[state.dealer_sum][end_v]
                if prob>0:
                    newState=State(dealer_sum=end_v,agent_sum=state.agent_sum)
                    reward=self.generate_reward_bust(newState)
                    ans.append((newState,prob,reward))
            bust_p=self.p_bust[state.dealer_sum]
            if bust_p>0:
                ans.append((State(dealer_sum=0,agent_sum=state.agent_sum,is_terminal=True),bust_p,1))
        elif action==Action.HIT.value:
            red,black=1/30,2/30
            a=state.agent_sum
            bust_p=max(a-11,0)*black+max(11-a,0)*red  # max{10-(22-a)+1,0}  max{10-a+1,0}
            ans.append((State(dealer_sum=state.dealer_sum,agent_sum=0,is_terminal=True),bust_p,-1))
            min_end=max(a-10,1)
            max_end=min(a+10,21)
            for v in range(min_end,state.agent_sum):
                ans.append((State(dealer_sum=state.dealer_sum, agent_sum=v), red, 0))
            for v in range(state.agent_sum,max_end+1):
                ans.append((State(dealer_sum=state.dealer_sum,agent_sum=v),black,0))
        return ans

    def step(self, s, a):
        """
            Given a state and an action return the next state.

            Args:
                s (State): Current state.
                a (Action): Action chosen by player.
            return:
                next_s (State): Next state
                r (Integer): Reward (-1, 0, 1).
        """
        # initially there's no reward and the next_s is equal to the
        # current state
        r = 0
        next_s = copy.copy(s)

        # if the player sticks then it's dealer turn
        if a == Action.STICK:
            next_s = self.dealer_turn(s)
            if next_s.is_terminal:
                r = 1
            else:
                next_s.is_terminal = True
                r = self.generate_reward_bust(next_s)
        else:
            next_s.agent_sum += self.take_card(self.deck)
            next_s.is_terminal = self.check_bust(next_s.agent_sum)

            # if end of the game then player lost: reward = -1
            if next_s.is_terminal:
                r = -1

        return next_s, r

environment = Environment()

class PolicyIterationAgent:
    def __init__(self,environment,discount):
        self.env=environment
        self.discount=discount
        self.V=np.zeros((self.env.dealer_max_value+1,self.env.agent_max_value+1))
        self.action=np.ones((self.env.dealer_max_value+1,self.env.agent_max_value+1))

    def getAction(self,state):
        return self.action[state.dealer_sum][state.agent_sum]

    def getV(self,state):
        if state.is_terminal or state.dealer_sum>self.env.dealer_max_value:
            return 0
        return self.V[state.dealer_sum][state.agent_sum]

    def get_value_function(self):
        return self.V

    def calculateV(self):
        eps=1e-3
        max_iter=10000
        iter=0
        while iter<max_iter:
            delta=0
            for dealer_sum in range(1,self.env.dealer_max_value+1):
                for agent_sum in range(1,self.env.agent_max_value+1):
                    v=0
                    state=State(dealer_sum=dealer_sum,agent_sum=agent_sum)
                    for new_state,prob,reward in self.env.getNewStateAndProbReward(state,self.getAction(state)):

                        v+=prob*(reward+self.getV(new_state))

                    delta=max(delta,abs(v-self.V[dealer_sum][agent_sum]))
                    self.V[dealer_sum][agent_sum]=v
            if delta<eps:
                break
        if iter==max_iter:
            print('reach max iteration times')

    def train(self,maxIter=100000):
        iter=0
        while iter<maxIter:
            iter+=1
            policy_stable=True
            self.calculateV()
            for dealer_sum in range(1,self.env.dealer_max_value+1):
                for agent_sum in range(1,self.env.agent_max_value+1):
                    v = np.zeros(2)
                    state = State(dealer_sum=dealer_sum, agent_sum=agent_sum)
                    for action in range(2):
                        for new_state, prob, reward in self.env.getNewStateAndProbReward(state, action):
                            v[action] += prob * (reward + self.getV(new_state))
                    policy=np.argmax(v)
                    if policy!=self.getAction(state):
                        policy_stable=False
                    self.action[dealer_sum][agent_sum]=policy
            if policy_stable:
                break
        print(f'number of iterations={iter}')
        if iter==maxIter:
            print('in training reach the max iteration times')

    def test(self,steps=100):
        '''估算胜率'''
        self.wins=0
        for e in range(steps):
            state=self.env.initial_state()
            reward=None
            while not state.is_terminal:
                action = Action.STICK if self.getAction(state)==0 else Action.HIT
                newstate,reward=self.env.step(copy.copy(state),action)
                state=newstate
            if reward == 1:
                self.wins += 1
        print(f'winning rate:{self.wins/steps},win times{self.wins}')


environment.endProb(simTimes=30000,verbose=False)
disc=1
poAgent=PolicyIterationAgent(environment,discount=disc)
poAgent.train()
poAgent.test(steps=100000)
plot_value_function(poAgent, title=f"Policy discount = {disc}")

'''disc,exploration=0.9,0.2
qlAgent=QLearningAgent(environment,discount=disc,explorationProb=exploration)
qlAgent.train(1000000)
plot_value_function(qlAgent, title=f"QL discount = {disc},explorationProb={exploration}")'''
