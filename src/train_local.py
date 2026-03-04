import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MS2CDataset
from model import MS2CModel
import os
import time  # For high-resolution step timing


def train():
    # 1. HARDWARE SETUP: Optimized for MacBook Pro 14 M1 Pro
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. DATA CONFIGURATION: Dynamic path resolution for robust execution
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    root_dir = os.path.join(project_root, "data")
    json_path = os.path.join(root_dir, "manifests", "spacing.json")

    # [MEMORY OPTIMIZATION]: Physical Batch Size = 1
    # This keeps Memory Pressure in the Green by only loading 1 sample into RAM at a time.
    dataset = MS2CDataset(json_path=json_path, root_dir=root_dir, category="spacing")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Effective batch size of 4 via accumulation
    accumulation_steps = 4

    # 3. MODEL INITIALIZATION
    model = MS2CModel().to(device)

    # 4. LOSS & OPTIMIZER
    # [LOGIC OPTIMIZATION]: Increased margin to 1.5 to force stronger feature separation
    criterion = nn.TripletMarginLoss(margin=1.5, p=2)
    consistency_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # 5. TRAINING LOOP
    model.train()
    print("Starting Optimized Spacing Module Training...")

    best_loss = float('inf')
    patience_limit = 3
    patience_counter = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Initial zero_grad for accumulation

        # Initialize timer for step logging
        step_start_time = time.time()

        for i, batch in enumerate(dataloader):
            # Move inputs to MPS
            anchor_img = batch['anchor_img'].to(device)
            seed_img = batch['seed_img'].to(device)

            anchor_text = {k: v.squeeze(1).to(device) for k, v in batch['anchor_text'].items()}
            pos_code = {k: v.squeeze(1).to(device) for k, v in batch['pos_code'].items()}
            neg_code = {k: v.squeeze(1).to(device) for k, v in batch['neg_code'].items()}
            seed_code = {k: v.squeeze(1).to(device) for k, v in batch['seed_code'].items()}

            # A. Forward Pass
            anchor_txt_feat, anchor_vis_feat, alpha = model(anchor_text, anchor_img, seed_img)
            anchor_combined = (alpha * anchor_txt_feat) + ((1 - alpha) * anchor_vis_feat)

            # B. Retrieval Targets
            with torch.no_grad():
                pos_feat = model.codebert(**pos_code).last_hidden_state[:, 0, :]
                pos_feat = torch.nn.functional.normalize(pos_feat, p=2, dim=1)
                neg_feat = model.codebert(**neg_code).last_hidden_state[:, 0, :]
                neg_feat = torch.nn.functional.normalize(neg_feat, p=2, dim=1)

            # C. Reference Consistency
            seed_outputs = model.codebert(**seed_code)
            seed_feat = seed_outputs.last_hidden_state[:, 0, :]
            seed_feat = torch.nn.functional.normalize(seed_feat, p=2, dim=1)

            # D. Loss Calculation (scaled by accumulation steps)
            triplet_loss = criterion(anchor_combined, pos_feat, neg_feat)
            ref_loss = consistency_criterion(pos_feat, seed_feat)
            loss = (triplet_loss + (0.1 * ref_loss)) / accumulation_steps

            # E. Backward Pass (Accumulates gradients)
            loss.backward()

            # F. Optimizer Step (Only every 4 steps)
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += (loss.item() * accumulation_steps)

            # [LOGGING]: Log every 20 steps with exact time spent per step
            if i % 20 == 0 and i > 0:
                step_end_time = time.time()
                elapsed_batch = step_end_time - step_start_time
                avg_time_per_step = elapsed_batch / 20

                print(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{i}], "
                    f"Loss: {loss.item() * accumulation_steps:.4f}, "
                    f"Alpha: {alpha.mean().item():.4f}, "
                    f"Time/Step: {avg_time_per_step:.2f}s"
                )

                # Reset timer for next 20 steps
                step_start_time = time.time()
            elif i == 0:
                # Special case for the very first step print
                print(
                    f"Epoch [{epoch + 1}/{max_epochs}], Step [{i}], "
                    f"Loss: {loss.item() * accumulation_steps:.4f}, "
                    f"Alpha: {alpha.mean().item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete. Average Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            checkpoint_dir = os.path.join(root_dir, "..", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "spacing_model_best.pth"))
            print(f"--> Saved New Best Model Checkpoint at Epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"EARLY STOPPING TRIGGERED at Epoch {epoch + 1}.")
                break


if __name__ == "__main__":
    train()
