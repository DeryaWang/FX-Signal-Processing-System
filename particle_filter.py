import numpy as np
import pandas as pd

class FXParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for FX Trend Extraction.
    Designed to handle non-Gaussian high-frequency data streams.
    """
    def __init__(self, n_particles=1000, process_vol=0.0005, measurement_vol=0.002):
        self.n_particles = n_particles
        self.process_vol = process_vol 
        self.measurement_vol = measurement_vol
        self.particles = np.zeros(n_particles)
        self.weights = np.ones(n_particles) / n_particles

    def predict(self):
        """
        Step 1: Prediction (State Transition)
        """
        noise = np.random.normal(0, self.process_vol, self.n_particles)
        self.particles += noise

    def update(self, observed_return):
        """
        Step 2: Update (Likelihood Calculation)
        Uses the Log-Likelihood trick for numerical stability.
        """
        # Calculate log-likelihood
        log_likelihood = -0.5 * ((observed_return - self.particles) / self.measurement_vol)**2
        
        # Stability shift
        log_likelihood -= np.max(log_likelihood)
        
        # Calculate new weights
        new_weights = np.exp(log_likelihood)
        
        # Multiply with existing weights
        self.weights *= new_weights
        
        # Normalize and handle potential underflow
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            # Safety valve: if all weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles

    def resample(self):
        """
        Step 3: Resampling
        Focus computational power on high-probability particles.
        Eliminates 'degeneracy' where only one particle has significant weight.
        """
        # Systematic Resampling
        indices = np.random.choice(np.arange(self.n_particles), size=self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def estimate(self):
        """
        Extract the current best estimate of the trend (weighted average).
        """
        return np.average(self.particles, weights=self.weights)

def run_particle_filter(returns_series):
    """
    Runs the filter over a sequence of log returns.
    """
    pf = FXParticleFilter(n_particles=1000)
    filtered_trend = []

    print("Running Particle Filter...")
    for obs in returns_series:
        pf.predict()
        pf.update(obs)
        # Resampling trigger (Effective Sample Size check)
        # For simplicity, we resample at every step in this version
        pf.resample()
        filtered_trend.append(pf.estimate())
    
    return np.array(filtered_trend)

if __name__ == "__main__":
    # Test script using previously saved data
    try:
        data = pd.read_csv("~/Desktop/FX/processed_data.csv")
        returns = data['Log_Returns'].values
        
        trend = run_particle_filter(returns)
        
        # Add the trend back to the dataframe
        data['Filtered_Trend'] = trend
        data.to_csv("~/Desktop/FX/filtered_data.csv", index=False)
        print(f"Filtering complete. Results saved with {len(trend)} trend points.")
    except Exception as e:
        print(f"Error: {e}. Did you run data_processor.py first?")
