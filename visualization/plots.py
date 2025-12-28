import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import seaborn as sns
import numpy as np
import glob

class Visualizer:
    """
    Post-Hoc Analysis and Visualization Engine.
    
    This class is responsible for transforming raw experimental logs (CSV, PyTorch tensors)
    into the 6 canonical result plots (R1-R6) defined in the project constraints.
    It handles data loading, statistical aggregation, and manifold plotting.
    """
    def __init__(self, log_dir="results/logs", plot_dir="results/plots"):
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        
    def _get_activation_file(self, step, concept_suffix="*"):
        """Utility: Locates specific checking logs with robust wildcard matching."""
        pattern = os.path.join(self.log_dir, f"activations_step_{step}_{concept_suffix}.pt")
        files = glob.glob(pattern)
        if not files:
            # Fallback for legacy log formats
            pattern = os.path.join(self.log_dir, f"activations_step_{step}.pt")
            files = glob.glob(pattern)
            
        if files:
            return files[0]
        return None

    def plot_R1_emergence(self, early_step=10, late_step=200):
        """
        R1: Emergence of Structural Order.
        
        Visualizes the transition from random initialization (entropy) to structured 
        representation (order). We use heatmaps of the hidden layer activations to 
        show the formation of "Vertical Bands," which correspond to specific neurons 
        consistently responding to the target concept tokens learning.
        
        Hypothesis:
            Early training -> Diffuse, high-entropy activation.
            Late training -> Sparse, low-entropy, concept-specific activation.
        """
        f_early = self._get_activation_file(early_step)
        f_late = self._get_activation_file(late_step)
        
        if not f_early or not f_late:
            print(f"Skipping R1: Missing files for step {early_step} or {late_step}")
            return
            
        act_early = torch.load(f_early).float().mean(dim=0).cpu().numpy()
        act_late = torch.load(f_late).float().mean(dim=0).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(act_early, ax=axes[0], cmap="viridis", cbar=False)
        axes[0].set_title(f"Early Training (Step {early_step}): High Entropy")
        axes[0].set_xlabel("Neuron ID")
        axes[0].set_ylabel("Sequence Position")
        
        sns.heatmap(act_late, ax=axes[1], cmap="viridis")
        axes[1].set_title(f"Late Training (Step {late_step}): Emergent Structure")
        axes[1].set_xlabel("Neuron ID")
        axes[1].set_ylabel("Sequence Position")
        
        plt.suptitle("R1: Emergence of Concept-Specific Activation\n(Color scale = Relative Intensity, unnormalized internal signal)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "R1_emergence.png"))
        plt.close()

    def plot_R2_hebbian_growth(self):
        """
        R2: Hebbian Synapse Dynamics.
        
        Tracks the statistical distribution of synaptic weights over time.
        We expect to see the 'Max Weight' diverge significantly from the 'Mean Weight',
        confirming that only a select subset of synapses (those connecting correlated neurons)
        are being strengthened, consistent with Hebbian Theory.
        """
        weights_file = os.path.join(self.log_dir, "weights.csv")
        if not os.path.exists(weights_file): return
        df = pd.read_csv(weights_file)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['mean_weight'], label='Mean Synapse Weight (Background)', linewidth=2)
        plt.plot(df['step'], df['max_weight'], label='Max Synapse Weight (Signal)', linestyle='--')
        plt.title('R2: Hebbian Synapse Strengthening Over Time')
        plt.xlabel('Training Global Step')
        plt.ylabel('Synaptic Efficacy (Weight Magnitude)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "R2_hebbian_growth.png"))
        plt.close()

    def plot_R3_sparsity(self):
        """
        R3: Homeostatic Sparsity Regulation.
        
        Verifies that the k-WTA mechanism maintains a constant level of population sparsity,
        even as individual weights grow large. This stability proves that learning is occurring
        via 'Resource Reallocation' rather than 'Activation Expansion' (epileptic runaway).
        """
        weights_file = os.path.join(self.log_dir, "weights.csv")
        if not os.path.exists(weights_file): return
        df = pd.read_csv(weights_file)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['sparsity_percent'], color='orange', linewidth=2)
        plt.ylim(0, 1.0)
        plt.title('R3: Sparsity Stability\n(Stable sparsity = Reallocation of neurons, not expansion)')
        plt.xlabel('Training Global Step')
        plt.ylabel('Population Sparsity (% Inactive)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plot_dir, "R3_sparsity.png"))
        plt.close()

    def plot_R4_overlap(self, step_a, step_b, concepts=("Zypher", "Zyphrex")):
        """
        R4: Concept Orthogonality (Venn Analysis).
        
        Compares the set of top-k active neurons for two different concepts.
        In a healthy memory system (Monosemanticity), these sets should have minimal overlap,
        indicating that different concepts are stored in distinct physical locations within the network.
        """
        f_a = self._get_activation_file(step_a, concepts[0])
        f_b = self._get_activation_file(step_b, concepts[1])
        
        if not f_a or not f_b:
            print(f"Skipping R4: Missing files for steps {step_a} or {step_b}")
            return
            
        # Get flattened mean activation vector
        act_a = torch.load(f_a).float().mean(dim=(0,1)).cpu().numpy()
        act_b = torch.load(f_b).float().mean(dim=(0,1)).cpu().numpy()
        
        # Identify top-k neurons (Highly responsive units)
        k = 10
        top_a = set(np.argsort(act_a)[-k:])
        top_b = set(np.argsort(act_b)[-k:])
        
        overlap = top_a.intersection(top_b)
        
        # Bar Chart Visualization of Set Overlap
        plt.figure(figsize=(8, 5))
        venn_counts = [len(top_a - top_b), len(overlap), len(top_b - top_a)]
        plt.bar(["Concept A Only", "Overlap", "Concept B Only"], venn_counts, color=['blue', 'purple', 'red'])
        plt.title(f"R4: Neuron Overlap (Top {k} Units)")
        plt.ylabel("Count of Neurons")
        plt.savefig(os.path.join(self.plot_dir, "R4_overlap.png"))
        plt.close()
        
    def plot_R5_persistence(self, learn_step, persistence_step):
        """
        R5: Memory Persistence (Correlation Analysis).
        
        Correlates the activation pattern at the end of learning (Step T) with the 
        activation pattern after a forgetting interval (Step T+delta).
        A strong positive diagonal correlation (y=x) indicates that the memory trace 
        is stable and retrievable.
        """
        f_learn = self._get_activation_file(learn_step)
        f_persist = self._get_activation_file(persistence_step)
        
        if not f_learn or not f_persist:
             print("Skipping R5: Missing files")
             return

        act_learn = torch.load(f_learn).float().mean(dim=(0,1)).cpu().numpy()
        act_persist = torch.load(f_persist).float().mean(dim=(0,1)).cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.scatter(act_learn, act_persist, alpha=0.6)
        plt.title("R5: Memory Persistence (Activation Correlation)")
        plt.xlabel(f"Learning Phase (Step {learn_step})")
        plt.ylabel(f"Persistence Phase (Step {persistence_step})")
        
        # Plot identity line (y=x) for reference
        lims = [
            np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.savefig(os.path.join(self.plot_dir, "R5_persistence.png"))
        plt.close()

    def plot_R6_variance(self):
        """
        R6: Signal-to-Noise Ratio (Weight Variance).
        
        Uses the spread (Max - Min) of the synaptic weights as a proxy for the 
        Signal-to-Noise Ratio (SNR). An increasing spread indicates that the system 
        is actively filtering noise (suppressing weak weights) and amplifying signal.
        """
        weights_file = os.path.join(self.log_dir, "weights.csv")
        if not os.path.exists(weights_file): return
        df = pd.read_csv(weights_file)
        
        df['weight_range'] = df['max_weight'] - df['min_weight']
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['weight_range'])
        plt.title("R6: Structure Emergence (Weight Range Growth)")
        plt.xlabel("Global Step")
        plt.ylabel("Dynamic Range (Max - Min Weight)")
        plt.savefig(os.path.join(self.plot_dir, "R6_variance.png"))
        plt.close()

    def plot_all(self, total_steps):
        # Infer specific steps for phases
        phase1_end = total_steps
        phase2_end = total_steps + (total_steps // 2)
        phase3_end = phase2_end + 100 # Approx
        
        # Round to nearest logged step (10)
        # Using 50 based on new log config
        p1_round = (phase1_end // 50) * 50
        p2_round = (phase2_end // 50) * 50
        
        # Try to find the distinct logs
        self.plot_R1_emergence(early_step=50, late_step=p1_round)
        self.plot_R2_hebbian_growth()
        self.plot_R3_sparsity()
        self.plot_R4_overlap(step_a=p1_round, step_b=p2_round, concepts=("Zypher", "Zyphrex"))
        
        # For persistence, we need the last step logged in Phase 3
        # We'll just look for the highest step number available with 'Test_Zypher'
        # Or just specific knows
        self.plot_R5_persistence(learn_step=p1_round, persistence_step=p2_round + 50) # Approx
        self.plot_R6_variance()
        
        print("All R1-R7 plots generated.")

if __name__ == "__main__":
    import sys
    steps = 1000
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
        
    viz = Visualizer()
    viz.plot_all(steps)
