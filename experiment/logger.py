import torch
import json
import os
import csv

class StateLogger:
    """
    Observability Module for the BDH Experiment.
    
    This class handles the systematic tracking of internal model states, specifically
    synaptic weights and neuron activations, to enable post-hoc analysis of memory formation.
    
    Logged Metrics:
        1. Weights (Time-Series): Mean, Max, Min, and Sparsity levels of the synapse matrix.
        2. Activations (Spatial): Per-neuron activation strength averaged over batches.
        3. Full State Snapshots: Periodic dumps of the entire weight matrix for manifold analysis.
    """
    def __init__(self, log_dir="results/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.activations_file = os.path.join(log_dir, "activations.csv")
        self.weights_file = os.path.join(log_dir, "weights.csv")
        
        # Initialize log files with schemas
        with open(self.activations_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Schema: Temporal step, Neuron Index (Space), Magnitude, Context Token, Semantic Concept Label
            writer.writerow(["step", "neuron_id", "activation_value", "token_in_batch", "concept"])
            
        with open(self.weights_file, 'w', newline='') as f:
            writer = csv.writer(f)
             # Schema: Temporal step, Global Weight Statistics
            writer.writerow(["step", "mean_weight", "max_weight", "min_weight", "sparsity_percent"])

    def log_step(self, step, model_state, activations, concept_name="Unknown"):
        """
        Records the internal state of the model at a specific training step.
        
        Args:
            step (int): The current global time-step of the experiment.
            model_state (BDHModel): The model instance (to access synaptic weights).
            activations (torch.Tensor): The sparse hidden states [batch, seq_len, hidden_dim]
            concept_name (str): Label of the dominant concept in the current batch for correlation.
        """
        # 1. Extract Synaptic Weight Statistics
        # We look at the connectivity matrix W to detect Hebbian growth.
        weights = model_state.memory.W.detach()
        mean_w = weights.mean().item()
        max_w = weights.max().item()
        min_w = weights.min().item()
        
        # 2. Calculate Activation Sparsity
        # Sparsity = 1 - Density. Essential for verifying the mechanism of 'resource reallocation'.
        non_zeros = (activations != 0).sum().item()
        total_elements = activations.numel()
        sparsity = 1.0 - (non_zeros / total_elements)
        
        with open(self.weights_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, mean_w, max_w, min_w, sparsity])
            
        # 3. Log Detailed State Snapshots
        # We perform heavy-weight logging (full tensors) at a lower frequency to manage I/O overhead.
        if step % 50 == 0:
            torch.save(weights, os.path.join(self.log_dir, f"weights_step_{step}.pt"))
            # Save activations with concept name to allow specific 'concept-probe' analysis later.
            torch.save(activations.detach(), os.path.join(self.log_dir, f"activations_step_{step}_{concept_name}.pt"))
            
            # 4. Log Spatially Resolved Activations
            # We average activation across the batch and sequence to get a 'neuron-level' activity map.
            # This data drives the R1 and R4 visualizations (Emergence and Overlap).
            avg_act = activations.detach().mean(dim=(0, 1)).cpu().tolist()
            with open(self.activations_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for neuron_idx, val in enumerate(avg_act):
                    if val > 0: # Sparse logging: only record active neurons
                         writer.writerow([step, neuron_idx, val, 0, concept_name])

    def log_concept_activation(self, step, concept_name, activations):
        """
        specific logger to track which neurons light up for a specific concept.
        """
        pass # To be implemented if we have explicit concept labels during logging
