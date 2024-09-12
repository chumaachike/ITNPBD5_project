import torch
import torch_xla.core.xla_model as xm

def train(model, train_loader, optimizer, arc_face, train_accuracy, device):
    model.train()
    running_loss = 0.0
    train_accuracy.reset()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)
        logits = arc_face(embeddings, labels)

        # Compute the cross-entropy loss
        cur_loss = torch.nn.functional.cross_entropy(logits, labels)

        if torch.isnan(cur_loss).any():
            print("NaN detected in loss, skipping batch")
            continue

        cur_loss.backward()
        xm.optimizer_step(optimizer)  # Use XLA-specific optimizer step

        running_loss += cur_loss.item()

        # Update accuracy metric
        preds = torch.argmax(logits, dim=1)
        train_accuracy.update(preds, labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = train_accuracy.compute().item()

    return epoch_loss, epoch_accuracy

def validate(model, val_loader, arc_face, val_accuracy, device):
    model.eval()
    val_loss = 0.0
    val_accuracy.reset()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_embeddings = model(images)
            logits = arc_face(val_embeddings, labels)
            cur_loss = torch.nn.functional.cross_entropy(logits, labels)
            val_loss += cur_loss.item()
            preds = torch.argmax(logits, dim=1)
            val_accuracy.update(preds, labels)

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_accuracy = val_accuracy.compute().item()

    return val_epoch_loss, val_epoch_accuracy

def train_tpu(model, train_accuracy, device, optimizer, arc_face, scheduler, train_loader, val_loader=None, val_accuracy=None, save_path="model.pth"):
    num_epochs = 75
    epoch_losses = []
    train_accuracies = []
    
    if val_loader:
        val_losses = []
        val_accuracies = []

    for epoch in range(num_epochs):
        # Train for one epoch
        epoch_loss, epoch_accuracy = train(
            model, train_loader, optimizer, arc_face, train_accuracy, device
        )

        if val_loader:
            # Validate for one epoch
            val_epoch_loss, val_epoch_accuracy = validate(
                model, val_loader, arc_face, val_accuracy, device
            )

        # Step the scheduler
        scheduler.step()

        # Log and print metrics
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        xm.master_print(f"Epoch {epoch+1}/{num_epochs}, Current LR: {current_lr:.6f}")
        xm.master_print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
        
        epoch_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        if val_loader:
            xm.master_print(f"Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}")
            val_losses.append(val_epoch_loss)
            val_accuracies.append(val_epoch_accuracy)

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model_file_path = f"{save_path}_epoch_{epoch+1}.pth"

            # Save model parameters, optimizer state, and ArcFace parameters
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'arcface_state_dict': arc_face.state_dict()  # Assuming arc_face is your ArcFace module
            }, model_file_path)

            xm.master_print(f"Model, optimizer, and ArcFace saved at {model_file_path}")

    if val_loader: 
        return epoch_losses, val_losses, train_accuracies, val_accuracies
    else:
        return epoch_losses, train_accuracies



def test(model, test_loader, arc_face, test_accuracy, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_accuracy.reset()

    with torch.no_grad():  # No need to track gradients during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            test_embeddings = model(images)
            logits = arc_face(test_embeddings, labels)
            
            # Compute the loss
            cur_loss = torch.nn.functional.cross_entropy(logits, labels)
            test_loss += cur_loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            test_accuracy.update(preds, labels)

    test_epoch_loss = test_loss / len(test_loader)
    test_epoch_accuracy = test_accuracy.compute().item()

    return test_epoch_loss, test_epoch_accuracy


