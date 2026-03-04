import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel


class MS2CModel(nn.Module):
    def __init__(self):
        super(MS2CModel, self).__init__()

        # 1. CORE ENCODERS: Handling the 'Multimodal Retrieval Gap'
        # CodeBERT encodes natural language and code segments
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        # Vision Transformer (ViT) processes UI screenshots as visual tokens
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # 2. GEOMETRIC FEATURE ALIGNMENT (MLP Head)
        # This trainable component maps visual features into code's semantic space
        # It is initialized with no pre-trained weights to learn mapping from scratch
        self.visual_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768)  # Stability for local MPS training
        )

        # 3. ADAPTIVE GATING NETWORK (The 'Decision Maker')
        # Calculates 'alpha' weight based on the predicted reliability of signals
        # Based on the Gated Multimodal Unit (GMU) logic
        self.gating_network = nn.Sequential(
            # Takes concatenated Text + Projected Visual Delta features
            nn.Linear(768 * 2, 1),
            nn.Sigmoid()  # Restricts alpha to [0, 1] range
        )

    def forward(self, text_input, buggy_img, seed_img=None):
        """
        Processes text and image streams to produce fused semantic vectors.
        [NEW LOGIC]: Accepts seed_img to calculate visual delta for gating.
        """
        # A. TEXT STREAM: Extract [CLS] token for global context
        text_outputs = self.codebert(**text_input)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # B. VISION STREAM: Process Buggy Screenshot
        buggy_outputs = self.vit(buggy_img)
        bug_vis_features = buggy_outputs.last_hidden_state[:, 0, :]

        # Project visual features into the Code Semantic Space
        projected_vision = self.visual_projection(bug_vis_features)

        # C. ADAPTIVE GATING: Calculate Delta-aware Alpha
        if seed_img is not None:
            # Encode Seed Image to establish baseline features
            seed_outputs = self.vit(seed_img)
            seed_vis_features = seed_outputs.last_hidden_state[:, 0, :]

            # [EDIT]: Visual Delta = Buggy Features - Seed Features
            # This identifies the specific UI defect blemish
            visual_delta = bug_vis_features - seed_vis_features
        else:
            visual_delta = bug_vis_features

        # [EDIT]: Align the visual delta to the code semantic space for the gating network
        # This ensures the gate compares "text meaning" vs "visual change meaning"
        projected_delta = self.visual_projection(visual_delta)

        # Normalize features to unit sphere for Cosine Similarity
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
        projected_vision = torch.nn.functional.normalize(projected_vision, p=2, dim=1)
        projected_delta = torch.nn.functional.normalize(projected_delta, p=2, dim=1)

        # [EDIT]: Use 'projected_delta' in the gate instead of 'projected_vision'
        # This resolves the "unused variable" error and implements true Delta-Gating logic
        gate_input = torch.cat((text_features, projected_delta), dim=1)
        alpha = self.gating_network(gate_input)

        return text_features, projected_vision, alpha
