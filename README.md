📚 Adaptive Learning using Deep Reinforcement Learning (DQN)

This project implements an AI-driven personalized teaching system using Deep Q-Learning (DQN).
The agent learns to teach a simulated student by selecting the right exercise (skill + difficulty) at each step to maximize long-term mastery.

The result is an adaptive learning system that automatically sequences exercises to rapidly improve the learner’s mastery across multiple skills.

🚀 Project Overview
Component	Purpose
env.py	Simulates a student learning environment (state, rewards, transitions)
agent.py	Deep Q-Network-based teaching agent
train.py	Trains the RL agent
evaluate.py	Tests trained model and generates mastery plot
mdp_states.py	Documentation and visualization of the MDP design
🎯 Goal of the Agent

Maximize the student's final mastery across 3 skills in the fewest number of steps.

🧠 How It Works
🏫 Student Environment

Each episode simulates a student with:

3 independent skills

Initial mastery values

Individual learning rates

Probability of guessing and slipping

Each action corresponds to giving an exercise:

(skill index, difficulty level)


Total actions = 6 (3 skills × 2 difficulty levels)

🤖 Reinforcement Learning Agent

The teaching agent uses Deep Q-Learning (DQN):

Fully connected neural network maps state → Q-values

Replay buffer for stable learning

Target network for stabilization

Epsilon-greedy exploration

The agent learns a teaching policy, not just how to maximize immediate scores.

📂 Folder / File Structure
│── agent.py
│── env.py
│── train.py
│── evaluate.py
│── mdp_states.py
│── dqn.pth                (generated after training)
│── mastery_plot.png       (generated after evaluation)
│── training_logs.pkl      (generated after training)

🔧 Installation
1️⃣ Clone the repository
git clone <your-repo-url>
cd <repo-folder>

2️⃣ Install dependencies
pip install torch numpy matplotlib


(Optional) For GPU training, install CUDA-enabled PyTorch from https://pytorch.org/

🏋️ Training the Agent

To train the model from scratch:

python train.py


Outputs:

Model saved → dqn.pth

Training logs saved → training_logs.pkl

Console summary showing mastery performance

🧪 Model Evaluation

After training, evaluate the agent with:

python evaluate.py


Outputs:

mastery_plot.png — Average mastery curves for the 3 skills

Evaluation statistics (rewards, mastery %, success rate, steps)

Example result plot (expected):

📈 All 3 skills gradually increase in mastery across evaluation episodes

🧬 Reward Design Summary
Component	Reward
Correct exercise response	+1.0
Incorrect response	−0.3
Mastery improvement	+12 × mastery_gain
Hard difficulty penalty	−0.05
Balanced mastery bonus	Reward when weakest skill > 0.7
Episode end bonus	+3 to +12 when high mastery achieved

This encourages:
✔ growth in mastery
✔ balanced progress across skills
✔ efficient teaching strategy

📊 State Space Structure

State (11-dim vector):

[ mastery(3), last_action(6), last_outcome(1) ]

🔥 Key Results

Agent learns to balance practice among skills

Hard questions used strategically for faster learning

Final average mastery > 0.9 in the evaluation phase

Demonstrates effective curriculum sequencing via RL

🛣 Future Improvements
Direction	Benefit
Add varying student types	Generalize across learners
LSTM-DQN	Learn long-term learning patterns
Multi-agent RL	Collaborative classroom simulation
UI dashboard	Live training progress visualization
Curriculum constraints	Real-world academic use
