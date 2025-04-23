import torch.nn as nn
import torch

class MoLFormerWithRegressionHead(nn.Module):    
    def __init__(self, model, input_dim = 768, output_dim = 1):
        super(MoLFormerWithRegressionHead, self).__init__()
        self.model = model
        self.linear_output = torch.nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'logits'):
            last_hidden_state = outputs.logits
        cls_output = last_hidden_state[:, 0, :]
        output = self.linear_output(cls_output)
        return output
        
