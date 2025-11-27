import torch

class FeedbackLoop:
    def __init__(self, model, feedback, merge_updater):
        self.model = model
        self.feedback = feedback
        self.merge_updater = merge_updater

    def step(self, batch):
        input_ids = batch["input_ids"]
        logits, attn = self.model(input_ids)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1)
        )

        grads = self.feedback.gradient_magnitudes(loss)
        entropy = self.feedback.attention_entropy(attn)
        token_losses = self.feedback.token_loss_contributions(logits, input_ids)

        scored = self.merge_updater.score_merges(grads, entropy, token_losses)
        new_merges, loss_val = scored, loss.item()

        return new_merges, loss_val
