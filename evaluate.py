import numpy as np
import matplotlib.pyplot as plt
from env import StudentEnv
from agent import DQNAgent
import os
import sys
def evaluate(model_path='dqn.pth', episodes=20, max_steps=50):
    env = StudentEnv(seed=123, max_steps=max_steps)
    agent = DQNAgent(env.state_size(), env.action_size())
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        print("The model file may be corrupted or incompatible.")
        sys.exit(1)
    
    agent.eps = 0.0 

    all_mastery_traces = []
    rewards = []
    mastery_reached_count = 0  

    for ep in range(episodes):
        state = env.reset()
        trace = [env.mastery.copy()]
        total_reward = 0.0

        for t in range(env.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            trace.append(env.mastery.copy())
            total_reward += reward
            state = next_state

            if done:
                break
        avg_mastery = np.mean(env.mastery)
        mastery_reached = avg_mastery > 0.95
        if mastery_reached:
            mastery_reached_count += 1
            mastery_reached_episodes.append(ep)
            print(f"Episode {ep}: Mastery goal reached! Avg Mastery: {avg_mastery:.3f}, Steps: {env.steps}")

        all_mastery_traces.append(np.array(trace))
        rewards.append(total_reward)

    if len(all_mastery_traces) == 0:
        print("No episodes completed. Cannot generate plot.")
        return
    
    if len(rewards) == 0:
        print("No rewards collected. Cannot compute statistics.")
        return
    
    try:
        max_len = max(t.shape[0] for t in all_mastery_traces)
        
        padded = []
        for t in all_mastery_traces:
            if t.shape[0] < max_len:
                pad = np.tile(t[-1], (max_len - t.shape[0], 1))
                padded.append(np.vstack([t, pad]))
            else:
                padded.append(t)
        padded = np.array(padded)
        avg = padded.mean(axis=0)
        steps = np.arange(avg.shape[0])
        plt.figure(figsize=(8, 5))
        for i in range(avg.shape[1]):
            plt.plot(steps, avg[:, i], label=f'Skill {i}')

        plt.xlabel('Step')
        plt.ylabel('Mastery')
        plt.title('Average Mastery Progress Across Evaluation Episodes')
        plt.legend()
        plt.grid(True)
        plt.savefig('mastery_plot.png')
        plt.close() 
        print("Saved mastery_plot.png")
    except Exception as e:
        print(f"Error generating plot: {e}")
    

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Episodes: {episodes}")
    print(f"Mastery > 0.95 Reached: {mastery_reached_count} times ({100*mastery_reached_count/episodes:.1f}%)")
    if mastery_reached_count > 0:
        print(f"Episodes that reached goal: {mastery_reached_episodes}")
    print(f"Average Reward per Episode: {np.mean(rewards):.2f}")
    print(f"Std Reward per Episode: {np.std(rewards):.2f}")
    final_masteries = [np.mean(trace[-1]) for trace in all_mastery_traces]
    print(f"Average Final Mastery: {np.mean(final_masteries):.3f}")
    print(f"Max Final Mastery: {np.max(final_masteries):.3f}")
    print(f"Min Final Mastery: {np.min(final_masteries):.3f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    evaluate()
