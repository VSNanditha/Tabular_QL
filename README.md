# Tabular_QL
Tabular Q-Learning techniques for two-agent domains using OpenAI gym libraries 
(Two versions of such domain available at https://github.com/VSNanditha/gym/tree/master/gym/envs/two_agents)

## Contents
### Action Selectors
&ensp;&ensp;&ensp;Action values are selected based on either exploitation of prior knowledge or Q-values or exploration. Weka Randomforest classifier is used as supervised learner. 
* Human Agent Transfer (HAT)
    
    Actions are taken from supervised learner devised using prior knowledge (in this case, human demonstrations) with a probability parameter which decays exponentially during the learning process. 
* Confidence based HAT (CHAT)
    
    Extended version of HAT, which is also a supervised learner devised using human demonstrations. Action values are exploited only when the confidence values are above a threshold.
* Coordination Confidence (CC)
    
    Confidence values for the joint actions of the joint agents are computed and action values are considered only when the joint confidence is above the threshold.
* Action selection by Exploration/Q-values
    
    If the selected supervised learner fails to suggest an action, either actions are selected based on Q - values or by exploration with certain exploration rate.
### Learner Classes
* Simple Tabular Q-Learner Class
    
    Learner class uses a simple Q-Learner function to store the knowledge gained.
* Dynamic Reuse of Prior Learner Class
    
    Learner class uses 3 different learner objects each instantiated with a Q table using CHAT, CC and learner without any supervisor.
    
## Usage

There are domain runners which call the learner classes over gym environments.
```
Step 1: env = gym.make("<Two-Agent gym domain>")
Step 2: Provide the parameters based on the description provided in each learner
```
* Learner without prior

    ```tabular_ql.train(<parameters>)```
* Learner with CHAT supervisor

    ```tabular_ql.chat_trainer(<parameters>)```
* Learner with CC supervisor

    ```tabular_ql.coord_chat_trainer(<parameters>)```
* Learner with DRoP supervisor

    ```tabular_ql.drop_trainer(<parameters>)```
    
* Generate human demonstrations
    
    ```tabular_ql.human_demonstration(<parameters>)```
    
* Learn from pickle-saved learner

    ```tabular_ql.pklload(<parameters>)```