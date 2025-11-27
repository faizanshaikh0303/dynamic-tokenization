import torch

class FeedbackSignals:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def attention_entropy(self, attn):
        if attn is None:
            return 0.0
        entropy = - (attn * torch.log(attn + 1e-9)).sum(dim=-1).mean().item()
        return entropy

    def gradient_magnitudes(self, loss):
        loss.backward(retain_graph=True)
        grads = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.abs().mean().item()

        self.model.zero_grad()
        return grads

    def token_loss_contributions(self, logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        losses = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return losses.detach().cpu().tolist()
