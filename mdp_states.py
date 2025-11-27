import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
from env import StudentEnv


class MDPStateAnalyzer:
    """Analyzer for MDP states and design in the Student Learning Environment."""
    
    def __init__(self, n_sks: int = 3, max_steps: int = 50):
        """
        Initialize the MDP state analyzer.
        
        Args:
            n_sks: Number of skills in the environment
            max_steps: Maximum steps per episode
        """
        self.n_sks = n_sks
        self.n_actions = n_sks * 2
        self.max_steps = max_steps
        self.env = StudentEnv(n_sks=n_sks, max_steps=max_steps)
    
    def get_state_components(self, state: np.ndarray) -> Dict[str, np.ndarray]:

        mastery = state[:self.n_sks]
        last_action = state[self.n_sks:self.n_sks + self.n_actions]
        last_outcome = state[-1]
        
        return {
            'mastery': mastery,
            'last_action': last_action,
            'last_outcome': last_outcome
        }
    
    def get_state_space_info(self) -> Dict:
        state_size = self.env.state_size()
        action_size = self.env.action_size()
        
        return {
            'state_dimension': state_size,
            'action_dimension': action_size,
            'state_components': {
                'mastery_dim': self.n_sks,
                'action_history_dim': self.n_actions,
                'outcome_dim': 1
            },
            'state_space_description': {
                'mastery_range': (0.0, 1.0),
                'action_history_type': 'one-hot',
                'outcome_range': (0.0, 1.0)
            },
            'action_space_description': {
                'total_actions': action_size,
                'actions_per_skill': 2,
                'action_encoding': 'skill_index * 2 + difficulty (0=easy, 1=hard)'
            }
        }
    
    def analyze_state_transitions(self, n_samples: int = 1000) -> Dict:
        transitions = []
        mastery_changes = []
        rewards = []
        
        for _ in range(n_samples):
            state = self.env.reset()
            action = self.env.sample_action()
            next_state, reward, done, info = self.env.step(action)
            
            state_comp = self.get_state_components(state)
            next_state_comp = self.get_state_components(next_state)
            
            mastery_change = next_state_comp['mastery'] - state_comp['mastery']
            mastery_changes.append(mastery_change)
            rewards.append(reward)
            
            transitions.append({
                'action': action,
                'reward': reward,
                'mastery_change': mastery_change,
                'correct': info.get('c', False)
            })
        
        mastery_changes = np.array(mastery_changes)
        rewards = np.array(rewards)
        
        return {
            'n_samples': n_samples,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_mastery_change': np.mean(mastery_changes, axis=0),
            'std_mastery_change': np.std(mastery_changes, axis=0),
            'max_mastery_change': np.max(mastery_changes, axis=0),
            'min_mastery_change': np.min(mastery_changes, axis=0)
        }
    
    def visualize_state_space(self, save_path: Optional[str] = 'mdp_state_analysis.png'):
        """
        Visualize the MDP state space structure.
        
        Args:
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sample states
        states = []
        masteries = []
        for _ in range(100):
            state = self.env.reset()
            states.append(state)
            comp = self.get_state_components(state)
            masteries.append(comp['mastery'])
        
        masteries = np.array(masteries)
        
        ax = axes[0, 0]
        for i in range(self.n_sks):
            ax.hist(masteries[:, i], alpha=0.6, label=f'Skill {i}', bins=20)
        ax.set_xlabel('Mastery Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Initial Mastery Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        ax = axes[0, 1]
        info = self.get_state_space_info()
        components = ['Mastery', 'Action History', 'Outcome']
        sizes = [
            info['state_components']['mastery_dim'],
            info['state_components']['action_history_dim'],
            info['state_components']['outcome_dim']
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(sizes, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('State Space Composition')
           # Plot 3: Transition analysis
        ax = axes[1, 0]
        transition_stats = self.analyze_state_transitions(n_samples=500)
        skills = [f'Skill {i}' for i in range(self.n_sks)]
        avg_changes = transition_stats['avg_mastery_change']
        std_changes = transition_stats['std_mastery_change']
        ax.bar(skills, avg_changes, yerr=std_changes, capsize=5, alpha=0.7)
        ax.set_ylabel('Average Mastery Change')
        ax.set_title('Average Mastery Change per Skill')
        ax.grid(True, alpha=0.3, axis='y')
        

        ax = axes[1, 1]
        action_counts = np.zeros(self.n_actions)
        for _ in range(1000):
            action = self.env.sample_action()
            action_counts[action] += 1
        action_labels = [f'S{i//2}-{"Hard" if i%2 else "Easy"}' 
                        for i in range(self.n_actions)]
        ax.bar(range(self.n_actions), action_counts, alpha=0.7)
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Space Distribution')
        ax.set_xticks(range(self.n_actions))
        ax.set_xticklabels(action_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        plt.close()
    
    def get_mdp_design_summary(self) -> str:
        """
        Get a text summary of the MDP design.
        
        Returns:
            Formatted string describing the MDP design
        """
        info = self.get_state_space_info()
        
        summary = f"""
================================================================================
MDP DESIGN SUMMARY - Student Learning Environment
================================================================================

STATE SPACE:
  - Total Dimension: {info['state_dimension']}
  - Components:
    * Mastery Levels: {info['state_components']['mastery_dim']} dimensions
      (Range: {info['state_space_description']['mastery_range']})
    * Action History: {info['state_components']['action_history_dim']} dimensions
      (Type: {info['state_space_description']['action_history_type']})
    * Last Outcome: {info['state_components']['outcome_dim']} dimension
      (Range: {info['state_space_description']['outcome_range']})

ACTION SPACE:
  - Total Actions: {info['action_dimension']}
  - Actions per Skill: {info['action_space_description']['actions_per_skill']}
  - Encoding: {info['action_space_description']['action_encoding']}
  - Action Types:
    * Easy difficulty (d=0): Standard mastery update
    * Hard difficulty (d=1): 75% base success rate, 1.5x learning rate

TRANSITION DYNAMICS:
  - Mastery updates based on:
    * Current mastery level
    * Learning rate (per skill)
    * Difficulty level (easy/hard)
    * Random noise factor (0.9-1.1)
  - Success probability: mastery * (1-slip) + (1-mastery) * guess
  - Mastery gain: learning_rate * (1 - old_mastery) * difficulty_factor * noise

REWARD STRUCTURE:
  - Correct answer: +1.0
  - Incorrect answer: -0.2
  - Mastery gain: +10.0 * mastery_gain
  - Hard difficulty penalty: -0.1

TERMINATION CONDITIONS:
  - Maximum steps reached: {self.max_steps}
  - Average mastery > 0.95

================================================================================
"""
        return summary


def print_mdp_design():
    """Print MDP design summary to console."""
    analyzer = MDPStateAnalyzer()
    print(analyzer.get_mdp_design_summary())


def analyze_mdp_states(n_samples: int = 1000):
    """
    Perform comprehensive MDP state analysis.
    
    Args:
        n_samples: Number of samples for transition analysis
    """
    analyzer = MDPStateAnalyzer()
    
    print("=" * 80)
    print("MDP STATE ANALYSIS")
    print("=" * 80)
    
    print("\n1. STATE SPACE INFORMATION:")
    info = analyzer.get_state_space_info()
    print(f"   State dimension: {info['state_dimension']}")
    print(f"   Action dimension: {info['action_dimension']}")
    print(f"   Mastery dimensions: {info['state_components']['mastery_dim']}")
    print(f"   Action history dimensions: {info['state_components']['action_history_dim']}")
       # Transition analysis
    print("\n2. TRANSITION ANALYSIS:")
    transition_stats = analyzer.analyze_state_transitions(n_samples=n_samples)
    print(f"   Average reward: {transition_stats['avg_reward']:.4f}")
    print(f"   Std reward: {transition_stats['std_reward']:.4f}")
    print(f"   Average mastery changes per skill:")
    for i, change in enumerate(transition_stats['avg_mastery_change']):
        print(f"     Skill {i}: {change:.6f} ± {transition_stats['std_mastery_change'][i]:.6f}")
    
    print("\n3. GENERATING VISUALIZATIONS...")
    analyzer.visualize_state_space()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    print_mdp_design()
    
    analyze_mdp_states(n_samples=1000)

