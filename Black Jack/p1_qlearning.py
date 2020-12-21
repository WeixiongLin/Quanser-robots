from p1_util import *
import numpy as np
import random, copy
from enum import Enum
import matplotlib.pyplot as plt
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

class Agent(Player):
    def __init__(self, environment, No=100, discount_factor=1):
        # Player is a superclass, which means an agent implements a policy
        Player.__init__(self)

        # easy21 environment
        self.env = environment

        # we can tune these parameters
        # don't worry about this for now
        self.No = No
        self.disc_factor = discount_factor

        # V(s) is the state value function. How good is to be at state s?
        # initially we don't know
        self.V = np.zeros([self.env.dealer_max_value + 1, self.env.agent_max_value + 1])

        # this will be used to we keep track of the agent's score
        # score is a simple metric to check if the agent is getting better over time
        # score = (# of wins)/(# of games played)
        self.wins = 0.0
        self.iterations = 0.0

    def get_clear_tensor(self):
        """This is just a helper function. Not important.

        Returns a tensor with zeros with the correct given shape for Q.
        By default this is (max possible dealer sum, max possible agent sum, number of actions)
        """
        return np.zeros((self.env.dealer_max_value + 1,
                         self.env.agent_max_value + 1,
                         self.env.actions_count))

    def choose_random_action(self):
        """We try to find the best policy possible but we act randomly sometimes."""
        return Action.HIT if random.random() <= 0.5 else Action.STICK

    def choose_best_action(self, s):
        """Returns the best action possible in state s."""
        raise NotImplemented()

    def get_max_action(self, s):
        """Returns the maxQ(s, a) between all actions."""
        return 0.0

    def get_value_function(self):
        """Get best value function in the moment."""
        for i in range(1, self.env.dealer_max_value + 1):
            for j in range(1, self.env.agent_max_value + 1):
                s = State(i, j)
                self.V[i][j] = self.get_max_action(s)
        return self.V

    def train(self, steps):
        """Train an agent for a certain number of steps.

           Args:
               steps (int): number of episodes to run.
           Returns:
               value function.
        """
        for e in range(steps):
            # do something...
            pass
        return self.get_value_function()

class QLearningAgent(Agent):
    def __init__(self,environment,discount=1,explorationProb=0.2):
        Agent.__init__(self,environment,100,discount)
        self.explorationProb=explorationProb
        self.Q=np.zeros((self.env.dealer_max_value + 1,self.env.agent_max_value + 1, self.env.actions_count))
        self.numIters=0
        self.actions=[Action.HIT,Action.STICK]
        self.N=self.get_clear_tensor()
    def getStepSize(self):
        return 1.0 / np.sqrt(self.numIters)
    def get_max_action(self, s):
        return np.max(self.Q[s.dealer_sum][s.agent_sum])
    def choose_best_action(self, s):
        return Action.HIT if np.argmax(self.Q[s.dealer_sum][s.agent_sum]) == 1 else Action.STICK
    def getQ(self,s,a):
        return self.Q[s.dealer_sum][s.agent_sum][a.value]
    def get_action(self,s):
        self.numIters += 1
        if random.random() < self.get_eps(s):
            action=self.choose_random_action()
        else:
            action=self.choose_best_action(s)
        self.N[s.dealer_sum][s.agent_sum][action.value]+=1
        return action
    def get_eps(self,state):
        return self.explorationProb
        #return self.No/((self.No + sum(self.N[state.dealer_sum, state.agent_sum, :]) * 1.0))
    def incorporateFeedback(self,state,action,reward,newState):
        V_opt = 0.0
        if newState is not None and not newState.is_terminal:
            V_opt = max([self.getQ(newState, newAction) for newAction in self.actions])
        Q_opt = self.getQ(state, action)
        adjustment = -self.getStepSize() * (Q_opt - (reward + self.disc_factor * V_opt))
        self.Q[state.dealer_sum][state.agent_sum][action.value]+=adjustment
    def train(self,steps):
        self.winRate=[]
        self.episodes=[]
        curve=[]
        for e in range(steps+1):
            state=self.env.initial_state()
            reward=0
            while not state.is_terminal:
                action = self.get_action(state)
                newstate,reward=self.env.step(copy.copy(state),action)
                self.incorporateFeedback(state,action,reward,newstate)
                state=newstate
            if e % 10000 == 0 and e!=0:
                score=float(self.wins) / self.iterations* 100
                print("Episode: %d, score: %f" % (e, score))
                self.winRate.append(score)
                self.episodes.append(e)
                curve.append(score)
            self.iterations += 1
            if reward == 1:
                self.wins += 1
        return curve
    
    def showAction(self):
        action=np.zeros((self.env.dealer_max_value+1,self.env.agent_max_value+1))
        for dealer_sum in range(1,self.env.dealer_max_value+1):
            for agent_sum in range(1,self.env.agent_max_value+1):
                s=State(dealer_sum,agent_sum)
                action[dealer_sum][agent_sum]=Action.HIT.value if np.argmax(self.Q[s.dealer_sum][s.agent_sum]) == 1 else Action.STICK.value
        return action
    
    def test(self,steps=100):
        '''估算胜率'''
        self.wins = 0
        for e in range(steps):
            state = self.env.initial_state()
            reward = None
            while not state.is_terminal:
                action = self.choose_best_action(state)
                newstate, reward = self.env.step(copy.copy(state), action)
                state = newstate
            if reward == 1:
                self.wins += 1
        print(f'winning rate:{self.wins / steps},win times{self.wins}')

disc=1
exp_rates=[0.1,0.2,0.3,0.4]
colors=['b','r','g','orange']
#steps=10000000
steps=1000000
exploration=0.1
qlAgent=QLearningAgent(environment,discount=disc,explorationProb=exploration)
winning_rate=qlAgent.train(steps)
plot_value_function(qlAgent, title=f"QL discount = {disc},explorationProb={exploration}")
qlAgent.test(int(1e5))

'''
episodes=[i for i in range(10000,steps+1,10000)]
exp_rates=[0.1,0.2,0.3,0.4]
colors=['b','r','g','orange']
steps=600000
for i,exp in enumerate(exp_rates):
    qlAgent = QLearningAgent(environment, discount=disc, explorationProb=exp)
    winning_rate = qlAgent.train(steps)
    plt.plot(episodes,winning_rate,color=colors[i],label=f'{exp}')
plt.xlabel('episodes')
plt.ylabel('accumulated winning rate')
plt.title('Effects of different exploration rates')
plt.legend()
plt.show()'''