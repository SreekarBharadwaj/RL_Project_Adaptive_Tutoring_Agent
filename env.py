# env.py
import numpy as np
class StudentEnv:
    """Simple adaptive learning environment with 3 sks."""

    def __init__(self, n_sks=3, max_steps=50, seed=None):
        self.n_sks = n_sks
        self.n_actions = n_sks * 2 
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self, student_profile=None):
        if student_profile is None:
            self.mastery = self.rng.uniform(0.2, 0.6, size=(self.n_sks,))
            self.learning_rate = self.rng.uniform(0.07, 0.18, size=(self.n_sks,))
            self.guess = 0.2
            self.slip = 0.05
        else:
            self.mastery = np.array(student_profile.get('mastery', [0.2]*self.n_sks))
            self.learning_rate = np.array(student_profile.get('learning_rate', [0.05]*self.n_sks))
            self.guess = student_profile.get('guess', 0.2)
            self.slip = student_profile.get('slip', 0.05)

        self.steps = 0
        self.last_action = np.zeros(self.n_actions)
        self.last_outcome = 0.0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.mastery, self.last_action, [self.last_outcome]]).astype(np.float32)

    def step(self, action):

        sk = action // 2
        d = action % 2

        bs = self.mastery[sk]
        if d == 1:
            bs *= 0.75

        pb = bs * (1 - self.slip) + (1 - bs) * self.guess
        c = self.rng.rand() < pb

        old_mastery = self.mastery[sk]
        if c:
            delta = self.learning_rate[sk] * (1.0 - old_mastery)
            if d == 1:
                delta *= 1.5
            difficulty_gain = 1.0 + 0.5 * (1.0 - np.mean(self.mastery))
            delta *= difficulty_gain
            delta *= (0.95 + 0.3 * self.rng.rand())
            self.mastery[sk] = min(1.0, old_mastery + delta)

        mastery_gain = self.mastery[sk] - old_mastery

        reward = 1.0 if c else -0.3
        reward += 12.0 * mastery_gain
        reward -= 0.05 * d

        avg_mastery = np.mean(self.mastery)
        weakest = np.min(self.mastery)
        reward += 3.0 * max(0.0, weakest - 0.7)
        if avg_mastery > 0.9:
            reward += 3.0
        if avg_mastery > 0.95:
            reward += 12.0

        self.last_action = np.zeros(self.n_actions)
        self.last_action[action] = 1.0
        self.last_outcome = 1.0 if c else 0.0

        self.steps += 1
        done = self.steps >= self.max_steps or np.mean(self.mastery) > 0.95

        return self._get_state(), reward, done, {'c': bool(c), 'mastery_gain': mastery_gain}

    def sample_action(self):
        return self.rng.randint(0, self.n_actions)

    def state_size(self):
        return self.n_sks + self.n_actions + 1

    def action_size(self):
        return self.n_actions
