class MergeUpdater:
    def __init__(self, merge_file):
        with open(merge_file) as f:
            self.merges = [
                line.strip()
                for line in f.readlines()
                if "#" not in line
            ]

    def score_merges(self, gradients, entropy, token_losses):
        scores = {}
        grad_total = sum(gradients.values())
        token_total = sum(token_losses)

        for merge in self.merges:
            scores[merge] = (
                entropy * 0.3 +
                grad_total * 0.5 +
                token_total * 0.2
            )
        return scores

    def update(self, scored_merges, k=50):
        sorted_merges = sorted(scored_merges.items(), key=lambda x: -x[1])
        return [m for m, _ in sorted_merges[:k]]
