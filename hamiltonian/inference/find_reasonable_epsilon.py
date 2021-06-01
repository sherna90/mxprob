import numpy as np

class DualAveragingStepSize:
    # https://colindcarroll.com/2019/04/21/step-size-adaptation-in-hamiltonian-monte-carlo/

    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10+initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        self.t += 1
        clip  = lambda val :  min(max(1e-5,val),0.9)
        return clip(np.exp(log_step)), clip(np.exp(self.log_averaged_step))
    
    def tune(self, acc_rate):
        if acc_rate < 0.001:
            # reduce by 90 percent
            self.mu *=0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            self.mu *= 0.5
        elif acc_rate < 0.2:
            self.mu *= 0.9
        elif acc_rate > 0.5:
            self.mu *= 1.1
        elif acc_rate > 0.75:
            self.mu *= 2
        elif acc_rate > 0.95:
            # increase by one thousand percent
            self.mu *= 10
        clip  = lambda val :  min(max(1e-5,val),0.9)
        return clip(np.exp(self.mu)),clip(np.exp(self.mu))