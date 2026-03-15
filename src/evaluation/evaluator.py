from .perplexity import compute_perplexity


class Evaluator:

    def __init__(self, model, dataloader, device):

        self.model = model
        self.dataloader = dataloader
        self.device = device

    def run(self):

        ppl = compute_perplexity(
            self.model,
            self.dataloader,
            self.device
        )

        return {
            "perplexity": ppl
        }