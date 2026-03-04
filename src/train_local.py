import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MS2CDataset  # Importing your 'Router' logic
from model import MS2CModel  # Importing the Gating & MLP logic
import os


def train():
    # 1. HARDWARE SETUP: Optimized for MacBook Pro 14 M1 Pro
    # We use 'mps' (Metal Performance Shaders) instead of 'cuda'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. DATA CONFIGURATION: Pointing to your 'Clean' folder structure
    # This aligns with the 'FULL project structure' we established
    # This looks for 'data' relative to the location of train_local.py itself
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    root_dir = os.path.join(project_root, "data")
    json_path = os.path.join(root_dir, "manifests", "spacing.json")

    # Initialize the Spacing Module dataset
    # [EDIT]: Dataset now fetches Seed Screenshots and Seed Code for reference
    dataset = MS2CDataset(json_path=json_path, root_dir=root_dir, category="spacing")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. MODEL INITIALIZATION
    model = MS2CModel().to(device)

    # 4. LOSS & OPTIMIZER: Implementing Triplet Protocol and AdamW
    # Triplet Loss pulls Anchor closer to Positive and pushes from Negative
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # [RESOLVED]: Added a simple MSE loss for structural consistency to utilize seed_feat
    consistency_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # [EDIT]: Added Learning Rate Scheduler to fine-tune weights on plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # 5. TRAINING LOOP
    model.train()
    print("Starting Spacing Module Training...")

    # [EDIT]: Changed to dynamic epochs with Early Stopping logic
    best_loss = float('inf')
    patience_limit = 3  # Maximum epochs to wait for improvement
    patience_counter = 0
    max_epochs = 50  # Upper limit; Early Stopping will likely trigger sooner

    for epoch in range(max_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # Move all inputs to MPS GPU
            anchor_img = batch['anchor_img'].to(device)
            # [EDIT]: Extracting Seed Screenshot for visual reference
            seed_img = batch['seed_img'].to(device)

            anchor_text = {k: v.squeeze(1).to(device) for k, v in batch['anchor_text'].items()}
            pos_code = {k: v.squeeze(1).to(device) for k, v in batch['pos_code'].items()}
            neg_code = {k: v.squeeze(1).to(device) for k, v in batch['neg_code'].items()}
            # [EDIT]: Extracting Seed Code (Fixed Version) for reference context
            seed_code = {k: v.squeeze(1).to(device) for k, v in batch['seed_code'].items()}

            optimizer.zero_grad()

            # A. Process the Anchor (Bug Report: Text + Image)
            # [EDIT]: Forward pass now compares anchor_img vs seed_img for alpha calculation
            anchor_txt_feat, anchor_vis_feat, alpha = model(anchor_text, anchor_img, seed_img)

            # B. Weighted Score Fusion for the Anchor
            # S_final = alpha * Text + (1-alpha) * Vision
            anchor_combined = (alpha * anchor_txt_feat) + ((1 - alpha) * anchor_vis_feat)

            # C. Process Positive and Negative Nodes through CodeBERT
            # We treat these as code-only embeddings for the retrieval target
            with torch.no_grad():
                # Positive: The actual buggy AST node
                pos_feat = model.codebert(**pos_code).last_hidden_state[:, 0, :]
                pos_feat = torch.nn.functional.normalize(pos_feat, p=2, dim=1)

                # Negative: Randomly sampled untampered node from the seed
                # This ensures the model learns the difference between buggy and healthy code
                neg_feat = model.codebert(**neg_code).last_hidden_state[:, 0, :]
                neg_feat = torch.nn.functional.normalize(neg_feat, p=2, dim=1)

            # [EDIT]: Optional Reference Consistency (Uses seed_code to ground the model)
            # This step helps the optimizer understand the global structure of the JSX file
            # even though it isn't the primary triplet loss target.
            seed_outputs = model.codebert(**seed_code)
            seed_feat = seed_outputs.last_hidden_state[:, 0, :]
            seed_feat = torch.nn.functional.normalize(seed_feat, p=2, dim=1)

            # D. Calculate Combined Loss
            # Main objective: Match the Anchor (Report) to the Positive (Buggy Node)
            triplet_loss = criterion(anchor_combined, pos_feat, neg_feat)

            # [RESOLVED]: Structural Consistency Loss (Uses the previously unused seed_feat)
            # Ensures the buggy node maintains a semantic relationship with its healthy origin
            ref_loss = consistency_criterion(pos_feat, seed_feat)

            # Final combined loss weighted for multi-task stability
            loss = triplet_loss + (0.1 * ref_loss)

            # E. Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{i}], Loss: {loss.item():.4f}, Alpha: {alpha.mean().item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        # [EDIT]: Step the scheduler and check for Early Stopping
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset counter on improvement

            # 6. SAVE CHECKPOINT: Only save if it is the best model so far
            checkpoint_dir = os.path.join(root_dir, "..", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            # [EDIT]: Use 'state_dict()' instead of the previous incorrect method name
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "spacing_model_best.pth"))
            print(f"--> Saved New Best Model Checkpoint at Epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"--> No improvement in loss for {patience_counter} epoch(s).")

            if patience_counter >= patience_limit:
                print(f"EARLY STOPPING TRIGGERED. Optimal convergence reached at Epoch {epoch + 1}.")
                break


if __name__ == "__main__":
    train()
