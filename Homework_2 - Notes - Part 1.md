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

## AI Output
- 
# General Notes
- Develop an intuition for the different parameters of a simple reinforcement learning agent
- We will have a problem called reward hacking
- iml/dungeon.py defines a simple gridworld environment, created by using maps generated in iml/dungeon maker.py
- Agent class in iml/agent.py defines the Q-learning agent (the knight) to be programmed externally
- DungeonVisualization class in iml/visualization.py provides the necessary plots to observe and interpret the agent’s strategies
- State --> Position in Dungeon
- Dungeon got walls "#" and got dragons "o". Dragon very bad very kill. Respawn at starting postion if dead
- Can take four possible integer actions {0,1,2,3}, each representing a direction in which the agent chooses to move
- Goal is to navigate from the starting position start (s) to the goal position end (e, representing the treasure) as ’optimally’ as possible
- Policy --> rule specifying which actions the agent should take as a (potentially stochastic) function of its current state
- For a given policy π(a|s), or simply π, let Vπ(i) be the expected return that an agent following the policy will receive per episode (running for a specific number of steps).
- Gamma --> Discount Factory --> Controls how much the agent values future rewards relative to immediate ones
- Meat states (+)

# Optional
We will do all bonuses if time allows

# Questions
- Slippery probability?
