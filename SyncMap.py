import torch

class SyncMap:
    def __init__(self, num_state: int, dimension: int):
        """SyncMap solver.

        Args:
            num_state: Number of states.
            dimension: Number of dimension for weight value.
        """        
        self.num_state = num_state
        self.dimension = dimension
        self.W = torch.rand(num_state, self.dimension)        

    def run(self, inputs):
        """Execute solver.

        Args:
            inputs: Inputs problem with size sequence_size*num_state.
        """        
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.W = self.W.to(device)
        inputs = inputs.to(device)
        positive = inputs > 0.1
        negative = ~positive

        pmass = torch.sum(positive, dim=1)
        nmass = torch.sum(negative, dim=1)

        for i in range(len(inputs)):
            if pmass[i] <= 1  or nmass[i] <= 1: continue

            center_pos = torch.sum(self.W[positive[i]], dim=0) / pmass[i]      # 1 * dimension
            center_neg = torch.sum(self.W[negative[i]], dim=0) / pmass[i]      # 1 * dimension
            center_pos = torch.unsqueeze(center_pos, 0)
            center_neg = torch.unsqueeze(center_neg, 0)

            dist_pos = torch.cdist(center_pos, self.W).reshape(self.num_state, 1)
            dist_neg = torch.cdist(center_neg, self.W).reshape(self.num_state, 1)

            phi = positive[i].repeat_interleave(self.dimension).reshape(self.num_state, self.dimension).long()
            update_pos = torch.mul(phi, center_pos - self.W) / dist_pos
            update_neg = torch.mul(1-phi, center_neg - self.W) / dist_neg
            update = update_pos - update_neg

            self.W += update
    
    def get_result(self):
        """Return the weight matrix.

        Returns:
            The weight matrix with size num_state*dimension.
        """        
        return self.W