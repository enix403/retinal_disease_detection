class Metrics:
    def __init__(self, dataset):
        tp = 0 # True positves
        fp = 0 # False positves
        tn = 0 # True negatives
        fn = 0 # True negatives

        total = 0

        dataloader = DataLoader(dataset, batch_size=64)

        with torch.no_grad():
            for images, labels in dataloader:
                logits = model(images)
                pred_labels = logits.argmax(dim=1)
                batch_size = pred_labels.shape[0]
                total += batch_size

                b_tp = (pred_labels * labels).sum()
                b_fp = (pred_labels * (1 - labels)).sum()
                b_tn = ((1 - pred_labels) * (1 - labels)).sum()
                b_fn = ((1 - pred_labels) * labels).sum()

                tp += b_tp
                fp += b_fp
                tn += b_tn
                fn += b_fn

        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        self.total = total


class Report:
    def __init__(self, metrics):
        self.metrics = metrics

    def accuracy(self):
        return (self.metrics.tp + self.metrics.tn) / self.metrics.total

    def precision(self):
        return self.metrics.tp / (self.metrics.tp + self.metrics.fp)

    def recall(self):
        return self.metrics.tp / (self.metrics.tp + self.metrics.fn)

