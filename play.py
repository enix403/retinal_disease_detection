for i in range(1):
    epoch_loss = 0
    num_batches = 0

    for images, labels in next(iter(train_dataloader)):
        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    print(f"Epoch {i+1}: train_loss={epoch_loss:.10f}")

print("Done")

"""
for i in range(1):
    epoch_loss = 0
    b = 0

    for images, labels in train_dataloader:
        model.train()

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, labels.view(-1, 1).float())
        print(f"{b+1}: {loss}")

        loss.backward()
        optimizer.step()

        # epoch_loss += loss.item()
        # num_batches += 1

    # print(f"==== Epoch {i+1}: train_loss={epoch_loss:.10f}")

print("Done")

"""


@torch.no_grad()
def calculate_loss(dataset):
    dataloader = DataLoader(dataset, batch_size=64)

    total_loss = 0
    for images, labels in dataloader:
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

    return total_loss