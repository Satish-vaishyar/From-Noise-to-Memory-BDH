import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    """
    Experimental Orchestrator.
    
    This class manages the lifecycle of the experiment, conducting the training loops
    for different phases (Acquisition, Control, Persistence) and interfacing between
    the data generator, the model, and the state logger.
    """
    def __init__(self, model, dataset, logger, learning_rate=0.01):
        self.model = model
        self.dataset_gen = dataset
        self.logger = logger
        # Standard Adam optimizer for any non-Hebbian components (projection layers).
        # Note: The Hebbian layer updates itself manually and is not stepped by this optimizer.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train(self, steps=1000, batch_size=32):
        """
        Phase 1: Primary Concept Acquisition.
        
        Trains the model on the primary concept ("Zypher") embedded in noise.
        This phase establishes the initial memory trace.
        """
        print("Starting training Phase 1 (Acquisition)...")
        self.model.train()
        
        for step in tqdm(range(1, steps + 1)):
            # 1. Generate Batch with probabilistic concept injection
            inputs, _ = self.dataset_gen.generate_data(concept_prob=0.3, concept_name="Zypher")
            
            # 2. Forward Pass with Hebbian Plasticity Enabled
            # The model internally updates its memory weights based on the input-output correlation.
            activations = self.model(inputs, update_hebbian=True)
            
            # 3. Optimization
            # (Optional in this pure-Hebbian demo, but kept for architectural completeness)
            # if model had backprop components, optimizer.step() would go here.
            
            # 4. Observability
            if step % 10 == 0:
                self.logger.log_step(step, self.model, activations, concept_name="Zypher")
                
        print("Training Phase 1 complete.")

    def run_control_experiment(self, steps=500, start_step=0):
        """
        Phase 2: Interference Control.
        
        Introduces a second, distinct concept ("Zyphrex") to the model.
        This allows us to verify:
        1. Whether the model can learn a new concept.
        2. Whether learning the new concept destroys the old one (Catastrophic Forgetting).
        """
        print("Running Control Experiment (Concept B: 'Zyphrex')...")
        
        for step in tqdm(range(1, steps + 1)):
            inputs, _ = self.dataset_gen.generate_data(concept_prob=0.3, concept_name="Zyphrex")
            
            # Plasticity enabled for new learning
            activations = self.model(inputs, update_hebbian=True)
            
            # Maintain global step count for continuous logging
            current_global_step = start_step + step
            
            if current_global_step % 10 == 0:
                self.logger.log_step(current_global_step, self.model, activations, concept_name="Zyphrex")

    def test_concept(self, concept_name="Zypher", steps=100, start_step=0):
        """
        Phase 3: Persistence/Recall Verification.
        
        Tests the model's response to a specific concept *without* further learning.
        This isolates the retrieval capability from the encoding capability.
        
        Args:
            update_hebbian (bool): Set to False to freeze memory state during testing.
        """
        print(f"Testing Concept Persistence ({concept_name})...")
        for step in tqdm(range(1, steps + 1)):
            # Force high probability of concept to guarantee distinct signal for measurement
            inputs, _ = self.dataset_gen.generate_data(concept_prob=1.0, concept_name=concept_name)
            
            # Disable Hebbian updates to test *retrieval* of existing memory structure
            activations = self.model(inputs, update_hebbian=False)
            
            current_global_step = start_step + step
            if current_global_step % 10 == 0:
                self.logger.log_step(current_global_step, self.model, activations, concept_name=f"Test_{concept_name}")
