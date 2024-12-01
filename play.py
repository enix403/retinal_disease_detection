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



num_display_img = 6
plt.figure(figsize=(12, 12))
_, axs = plt.subplots(num_display_img / 2, 3,
                 figsize=(6, 2 * num_display_img),
                 sharex=True, sharey=True)

for i in range(sample_count):
    plt.subplot(5, 2, i + 1)
    plt.imshow(images[i])
    plt.title(f"True: {true_labels[i]}, Pred: {pred_labels[i][0]}")
    plt.axis('off')
plt.show()


"""
_, axs = plt.subplots(num_display_img, 3,,
                 figsize=(6, 2 * num_display_img),,
                 sharex=True, sharey=True),
,
for i in range(num_display_img):,
 image = images[i],
 noised_image = noised_images[i],
 denoised_image = denoised_images[i],
,
 axs[i, 0].imshow(image.movedim(0,-1)),
 axs[i, 1].imshow(noised_image.movedim(0,-1)),
 axs[i, 2].imshow(denoised_image.movedim(0,-1)),
,
axs[0, 0].set_title(Original),
axs[0, 1].set_title(Noised),
axs[0, 2].set_title(Denoised)

"""