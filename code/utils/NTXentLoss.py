import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR style).
    No class labels required. It relies strictly on matching corresponding augmented views.
    """
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        # We use standard CrossEntropyLoss under the hood to handle the log-softmax math
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, proj_1, proj_2):
        """
        Args:
            proj_1: Normalized projections of View 1 [batch_size, projection_dim]
            proj_2: Normalized projections of View 2 [batch_size, projection_dim]
        """
        # FORCE autocast off for this entire block to prevent Float16 overflow!
        with torch.amp.autocast("cuda", enabled=False):
            
            # 0. Ensure inputs are strictly Float32 right at the start
            proj_1 = proj_1.float()
            proj_2 = proj_2.float()

            batch_size = proj_1.shape[0]
            device = proj_1.device
            
            # 1. Concatenate both views to form a single giant batch of size 2N
            out = torch.cat([proj_1, proj_2], dim=0) # Shape: [2 * batch_size, projection_dim]
            
            # 2. Compute the complete similarity matrix via matrix multiplication: [2N x 2N]
            # Since projections are already L2-normalized, this calculates Cosine Similarity
            sim_matrix = torch.matmul(out, out.T) / self.temperature
            
            # 3. Create a mask to block out self-similarity along the diagonal
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
            sim_matrix = sim_matrix.masked_fill(mask, -9e15) # Safely flood diagonal in Float32
            
            # 4. Construct the ground-truth target indices for the cross-entropy calculation
            # For an item i in View 1 (index 0 to N-1), its positive match sits at index i + N
            # For an item i in View 2 (index N to 2N-1), its positive match sits at index i - N
            targets = torch.arange(batch_size, device=device)
            targets = torch.cat([targets + batch_size, targets], dim=0) # Shape: [2 * batch_size]
            
            # 5. Compute the loss
            loss = self.criterion(sim_matrix, targets)
            
            return loss