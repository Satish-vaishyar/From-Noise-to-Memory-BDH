import random
import torch

class SyntheticDataset:
    """
    Synthetic Data Generator for BDH Concept Acquisition Experiments.
    
    This class generates sequences of integer tokens simulating a stream of data.
    It injects specific synthetic concepts ("Zypher", "Zyphrex") into the data stream
    with controllable probability, allowing us to strictly control the learning environment
    and isolate the "noise-to-memory" transition.
    
    Attributes:
        sequence_length (int): The fixed length of each generated sequence.
        vocab_size (int): The size of the vocabulary (integer token space).
        num_sequences (int): The number of sequences to generate per batch.
        concepts (dict): A mapping of concept names to their reserved unique token IDs.
    """
    def __init__(self, sequence_length=10, vocab_size=50, num_sequences=1000):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_sequences = num_sequences
        
        # Reserved Token IDs for specific concepts to track.
        # IDs 1 and 2 are reserved for our target concepts.
        # IDs 3+ are used for random "noise" tokens.
        self.concepts = {
            "Zypher": 1,  # Primary Experimental Concept
            "Zyphrex": 2   # Control Concept
        }
        
    def generate_data(self, concept_prob=0.1, concept_name="Zypher"):
        """
        Generates a batch of synthetic sequences with conditional concept injection.
        
        Args:
            concept_prob (float): The probability (0.0 to 1.0) that a given sequence 
                                  confirms the target concept.
            concept_name (str): The specific concept token to inject (e.g., "Zypher").
            
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, sequence_length) containing token IDs.
            torch.Tensor: A binary label tensor of shape (num_sequences,) indicating concept presence.
        """
        target_token = self.concepts.get(concept_name, 1)
        data = []
        labels = [] # Binary indicator: 1 if concept is present, 0 otherwise.
        
        for _ in range(self.num_sequences):
            # initialize sequence with random "background noise" tokens.
            # We sample from [3, vocab_size) to avoid colliding with reserved concept IDs.
            seq = [random.randint(3, self.vocab_size - 1) for _ in range(self.sequence_length)]
            
            label = 0
            # Probabilistic injection of the target concept
            if random.random() < concept_prob:
                # Insert the concept token at a random position within the sequence.
                # This breaks positional dependency, forcing the model to learn the token identity itself.
                pos = random.randint(0, self.sequence_length - 1)
                seq[pos] = target_token
                label = 1
                
            data.append(seq)
            labels.append(label)
            
        return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

if __name__ == "__main__":
    # Integration Test: Verify data generation logic
    gen = SyntheticDataset()
    data, labels = gen.generate_data(concept_prob=0.5, concept_name="Zypher")
    print(f"Generated data shape: {data.shape}")
    print(f"Example sequence: {data[0]}")
    print(f"Label: {labels[0]}")
