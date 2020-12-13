# IPM-MovePlanner
This repo is our IERG 5350 project about 'IPM Move Planner: An Efficient Exploiting Deep Reinforcement Learning with Monte Carlo Tree Search'
# Introduction
We have implemented an efficient IPM-Move Planner, with specific contributions as follows:
1. Use PPO to learn non-deterministic policy through probability.
2. Directly pre-training the policy network through supervised imitation learning, and then using PPO to train the policy value network to learn advanced policy, thus achieving the purpose of learning complex policy only with a lightweight network.
3. Combined the lightweight policy value network with MCTS to achieve an efficient long-term decision model.

![image](https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/Structure_diagram.png)

# Environment Installation
1. Install Requirements
  python 3.6
  pytorch 1.7.1
  CUDA 10.1
  tensorflow 1.15.0
2. Build Files 
```
  cd src/utils
  g++ -shared -O2 search.cpp --std=c++11 -ldl -fPIC -o search.so
```
The configuration method of these environments can refer to the [link](https://github.com/HanqingWangAI/SceneMover)

# Experimental results
## Compare network structure
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/net_success_rate.jpg" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/net_loss.jpg" width="400" height="270" />

## Compare PPO with PPO+IL
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/rewards.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/test_average_step.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/test_success_rate.png" width="400" height="270" />
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/loss.png" width="400" height="270" />

## Compare our method with others
.<img src="https://github.com/baifanxxx/IPM-MovePlanner/blob/main/IPM-MovePlaner/figs/table.jpg"/>
<!--  
>### Remark
>Part of the code in this project refers to [SceneMover](https://github.com/HanqingWangAI/SceneMover), if you use the code of this project, please refer to this project and >[SceneMover](https://github.com/HanqingWangAI/SceneMover)
--> 
