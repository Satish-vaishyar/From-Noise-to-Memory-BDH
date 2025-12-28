import argparse
import sys
import os

# Ensure the project root is in the python path for module resolution
sys.path.append(os.getcwd())

from data.generator import SyntheticDataset
from models.bdh import BDHModel
from experiment.logger import StateLogger
from experiment.trainer import Trainer
from visualization.plots import Visualizer

def main():
    """
    Main Entry Point for the BDH Experiment.
    
    This script orchestrates the full experimental pipeline:
    1.  **Data Initialization**: Sets up the synthetic concept stream generator.
    2.  **Model Initialization**: Instantiates the BDH architecture with Hebbian memory.
    3.  **Observability**: Configures logging for weights and activations.
    4.  **Experimental Phases**: Runs the 3-stage protocol (Acquisition -> Interference -> Persistence).
    5.  **Analysis**: Generates the R1-R7 visualizations from the logged data.
    """
    parser = argparse.ArgumentParser(description="BDH Memory Visualization Experiment Framework")
    parser.add_argument("--steps", type=int, default=200, help="Duration of the primary learning phase (Phase 1).")
    parser.add_argument("--test", action="store_true", help="Execute a minimal integration test to verify pipeline integrity.")
    parser.add_argument("--vocab_size", type=int, default=50, help="Size of the synthetic token vocabulary.")
    args = parser.parse_args()

    # 1. Setup Data Generator
    print("Initializing Synthetic Dataset...")
    dataset = SyntheticDataset(vocab_size=args.vocab_size)

    # 2. Setup Model Architecture
    print("Initializing BDH Model...")
    # Capacity Configuration:
    # d_model=64 / hidden_dim=128 selected empirically to ensure sufficient orthogonality 
    # for concept separation (addressing R4: Concept Isolation).
    model = BDHModel(vocab_size=args.vocab_size, embedding_dim=64, hidden_dim=128, sparsity_k=5)

    # 3. Setup Internal State Logger
    print("Initializing Observability Module...")
    logger = StateLogger(log_dir="results/logs")

    # 4. Setup Training Orchestrator
    print("Initializing Experimental Orchestrator...")
    trainer = Trainer(model, dataset, logger)

    # 5. Execute Experimental Pipeline
    if args.test:
        print("Running in TEST mode (Integration Check)...")
        trainer.train(steps=10)
    else:
        # Phase 1: Primary Acquisition ("Zypher")
        # Objective: Demonstrate the emergence of stable memory structures from noise.
        print("--- Phase 1: Learning Concept A (Zypher) ---")
        trainer.train(steps=args.steps)
        
        # Phase 2: Interference Control ("Zyphrex")
        # Objective: Demonstrate that new concepts can be learned without overwriting 
        # the old concept's specific neurons (Plasticity-Stability dilemma).
        print("--- Phase 2: Control Concept B (Zyphrex) ---")
        trainer.run_control_experiment(steps=args.steps // 2, start_step=args.steps)
        
        # Phase 3: Persistence / Recall Verification ("Zypher")
        # Objective: Verify that the original concept trace remains retrievable 
        # even after the model has processed unrelated data.
        print("--- Phase 3: Persistence Check (Zypher) ---")
        current_step = args.steps + (args.steps // 2)
        trainer.test_concept(concept_name="Zypher", steps=100, start_step=current_step)

    # 6. Post-Hoc Analysis & Visualization
    print("Generating R1-R7 Visualizations...")
    viz = Visualizer(log_dir="results/logs", plot_dir="results/plots")
    viz.plot_all(total_steps=args.steps)
    
    print("Experiment Complete. Results generated in 'results/plots/'.")

if __name__ == "__main__":
    main()
