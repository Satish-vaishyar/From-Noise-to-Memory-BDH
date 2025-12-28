import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLayer(nn.Module):
    """
    Implements a Hebbian Learning Layer that evolves via local synaptic plasticity rules.
    
    unlike standard backpropagation-trained layers, this layer updates its weights 
    during the forward pass based on the correlation between pre-synaptic inputs (x) 
    and post-synaptic outputs (y).
    
    Mechanism:
        Delta_W_ij = eta * y_i * x_j
        
    Where:
        eta: Learning rate (plasticity coefficient)
        y_i: Activation of post-synaptic neuron i
        x_j: Activation of pre-synaptic neuron j
    """
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        super(HebbianLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Plastic weights: W_hebb
        # Initialized with small random values to break symmetry, analogous to weak initial synaptic connections.
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        
    def forward(self, x):
        """
        Computes the linear transformation y = Wx.
        """
        return F.linear(x, self.W)
    
    def update_weights(self, x, y):
        """
        Applies the Oja-style / Hebbian update rule to the synaptic weights.
        
        Args:
            x (torch.Tensor): Pre-synaptic activations [batch, input_dim]
            y (torch.Tensor): Post-synaptic activations [batch, output_dim]
            
        Note:
            We use torch.no_grad() because this update is "biological" and manual,
            not derived from a global loss function via autograd.
        """
        
        with torch.no_grad():
            # Compute the outer product for each example in the batch to get discrete weight updates.
            # delta shape: [batch, output_dim, input_dim]
            # equivalent to: y column vector * x row vector
            delta = torch.bmm(y.unsqueeze(2), x.unsqueeze(1)) 
            
            # Average the updates over the batch to get a stable mean update steps.
            mean_delta = delta.mean(dim=0)
            
            # Apply the update: W_new = W_old + eta * Delta_W
            self.W += self.learning_rate * mean_delta
            
            # Weight Decay / Normalization:
            # Prevents runaway feedback loops where weights grow infinitely.
            # This is analogous to homeostatic plasticity in biological systems.
            self.W -= 0.001 * self.W 

class BDHModel(nn.Module):
    """
    Biological Dragon Hatchling (BDH) Model Architecture.
    
    This model integrates a Hebbian Memory Layer with a k-Winner-Take-All (k-WTA) 
    sparsity mechanism to simulate rapid, interference-free learning.
    
    Architecture:
        1. Embedding Layer: Converts discrete tokens to dense vectors.
        2. Projection Layer: Maps embeddings to the hidden dimension.
        3. Hebbian Memory: The core plastic layer that stores associative memories.
        4. Sparsity Enforcement: Selects the top-k most active neurons to maintain representation orthogonality.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sparsity_k=5):
        super(BDHModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Projection to the hidden state space
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # The core "Memory" layer where learning occurs
        self.memory = HebbianLayer(hidden_dim, hidden_dim)
        
        self.sparsity_k = sparsity_k
        self.hidden_dim = hidden_dim
        # Note: Output layer is handled implicitly for visualization purposes in this demo (direct activation observation)
        # In a full language model, a decoding layer would exist here.
        
        # Output decoding layer (added for completeness/compatibility)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, update_hebbian=True):
        """
        Forward pass with optional plastic updates.
        
        Args:
            x (torch.Tensor): Input token indices [batch, seq_len]
            update_hebbian (bool): Whether to trigger synaptic plasticity updates during this pass.
            
        Returns:
            torch.Tensor: Sparse activations of the memory layer [batch, seq_len, hidden_dim]
        """
        # Embed: [batch, seq_len, emb_dim]
        x_emb = self.embedding(x)
        
        batch_size, seq_len, _ = x_emb.shape
        
        # Flatten token sequence for unified processing: [batch * seq_len, emb_dim]
        x_flat = x_emb.view(-1, x_emb.size(2))
        
        # Project inputs to hidden dimension
        # ReLU ensures non-negative input signals, which is biologically plausible (firing rates >= 0)
        h_in = F.relu(self.input_proj(x_flat))
        
        # Hebbian Memory Pass
        # h_mem = W * h_in
        h_mem = self.memory(h_in)
        
        # Sparsity Enforcement (k-Winner-Take-All)
        # We retain only the 'k' most active neurons and suppress the rest to 0.
        # This forces the network to use sparse distributed representations (SDRs).
        top_k_vals, _ = torch.topk(h_mem, self.sparsity_k, dim=1)
        
        # Determine the activation threshold for each sample (the k-th largest value)
        kth_val = top_k_vals[:, -1].unsqueeze(1)
        
        # Create binary mask: 1 if activation >= threshold, else 0
        mask = (h_mem >= kth_val).float()
        
        # Apply mask: "The winner takes it all" (or k winners take it)
        h_sparse = h_mem * mask
        
        # Plasticity Step:
        # If we are in a learning phase, update the memory weights based on 
        # the correlation between the input (h_in) and the sparse output (h_sparse).
        if update_hebbian:
            self.memory.update_weights(h_in, h_sparse)
            
        # Compute output logits (for completeness, though primary analysis is on h_sparse)
        logits = self.output_layer(h_sparse)
            
        # Return logits and internal activations for visualization
        return logits, h_sparse.view(batch_size, seq_len, self.hidden_dim)

if __name__ == "__main__":
    # Unit Test: Architecture Compatibility
    model = BDHModel(vocab_size=50, embedding_dim=64, hidden_dim=128)
    fake_input = torch.randint(0, 50, (4, 10)) # batch 4, len 10
    logits, activations = model(fake_input)
    print("Logits shape:", logits.shape)
    print("Activations shape:", activations.shape)
    print("Memory weights mean:", model.memory.W.mean().item())
