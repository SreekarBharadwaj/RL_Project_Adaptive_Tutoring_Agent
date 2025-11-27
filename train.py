import numpy as np
from env import StudentEnv
from agent import DQNAgent
def train(episodes=500, max_steps=50):
    env = StudentEnv(seed=42, max_steps=max_steps)
    state_dim = env.state_size()
    action_dim = env.action_size()
    agent = DQNAgent(state_dim, action_dim)
    losses = []
    rewards_history = []
    final_masteries = [] 
    mastery_reached_count = 0  
    mastery_reached_episodes = []  

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size=128)
            state = next_state
            total_reward += reward
            if done:
                break

        avg_mastery = np.mean(env.mastery)
        final_masteries.append(avg_mastery)
        mastery_reached = avg_mastery > 0.95
        if mastery_reached:
            mastery_reached_count += 1
            mastery_reached_episodes.append(ep)

        if ep % 5 == 0:
            agent.sync_target()
        
        losses.append(loss if loss > 0 else 0.0)
        rewards_history.append(total_reward)
        
        if ep % 20 == 0:
            goal_status = "[GOAL]" if mastery_reached else ""
            print(f"Episode {ep:04d} | TotalReward: {total_reward:.2f} | AvgMastery: {avg_mastery:.3f} | Eps: {agent.eps:.3f} | Steps: {env.steps} {goal_status}")
    

    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Episodes: {episodes}")
    print(f"Mastery > 0.95 Reached: {mastery_reached_count} times ({100*mastery_reached_count/episodes:.1f}%)")
    if mastery_reached_count > 0:
        print(f"First Goal Reached: Episode {mastery_reached_episodes[0]}")
        print(f"Last Goal Reached: Episode {mastery_reached_episodes[-1]}")
    print(f"Average Final Reward: {np.mean(rewards_history):.2f}")
    print(f"Average Final Mastery: {np.mean(final_masteries):.3f}")
    print(f"Max Final Mastery: {np.max(final_masteries):.3f}")
    print(f"Min Final Mastery: {np.min(final_masteries):.3f}")
    print(f"{'='*70}\n")
    
    try:
        agent.save('dqn.pth')
        print(f"\nModel saved to 'dqn.pth'")
    except Exception as e:
        print(f"Error saving model: {e}")
    

    try:
        import pickle
        with open('training_logs.pkl', 'wb') as f:
            pickle.dump({'losses': losses, 'rewards': rewards_history}, f)
        print("Training logs saved to 'training_logs.pkl'")
    except Exception as e:
        print(f"Error saving training logs: {e}")


if __name__ == '__main__':
    train(episodes=500, max_steps=50)
