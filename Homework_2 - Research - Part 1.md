# Subjects required to do TP
- Q-learning
    - Episode length
    - Q-table
    - ε-greed
    - Discounted rewards?
    - Q-learning Agents
    - States
    - Policy
- Markov decision process (Lecture 7, Slide 35 (40/48))
- Exercise (c)
    - Theoretical probability distribution
    - Marginalizing
- Bellman expectation equation

# Research
## Links
- Q-learning
    - https://www.youtube.com/watch?v=RiSCh4H-GY4
    - https://www.youtube.com/watch?v=aCEvtRtNO-M
    - https://www.youtube.com/watch?v=OJcTnFw0oGE
- Markov Decision Process (MDP)
    - https://www.youtube.com/watch?v=4Fqt2Nk2lhY
    - https://www.youtube.com/watch?v=2iF9PRriA7w

## Q-learning
- Maximize reward
- Q-Values
    - Estimations of the current and future reward an agent expacts to receive
    - Q-Learning formula: Q(st,a) = rt + gamma * max (Q(st+1,a')) (More or less this)
        - Where:
            - s --> State (The current state before an action)
            - a --> Action (The action after state)
            - r --> Reward (Reward 1 or -1)
            - gamme --> Discount factor (0, present > future / 1, fut > pre)
    - Q-Table
        - States vs Action table where each value is a reward of performing an action in said state
        - Usually defaulted by having 0's on all?
        - Created by performing actions in certain states, and checking what the reward is
    - ε-greed --> How often does an agent move at random (e.g. 0.1 is 1 in 10 moves random)
    - Model-Free learning --> Predict value function of a policy without having a concrete model of the environment
        - Q-Learning IS model-free.
        - Q-Learning is a model-free technique that can be used to find the optimal action-selection policy using a Q-Function
- Bellman equation --> Is just the Q-Learning formula? (Or not, might actually be Markov)
- Markov Decision Process (MDP)
    - Policy --> Recipe which tells you what to do in a state
