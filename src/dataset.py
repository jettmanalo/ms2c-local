import os
import json
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
# [EDIT]: Importing the custom ASTParser for syntactic sanitization
from utils.ast_parser import ASTParser


class MS2CDataset(Dataset):
    def __init__(self, json_path, root_dir, category="spacing"):
        """
        Custom Dataset for M-S2C Parallel Retrieval.
        Routes mutation IDs to specific seed and buggy files for comparative learning.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir
        self.category = category

        # Initialize Encoders
        # CodeBERT handles natural language and AST nodes
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        # ViT processor handles image resizing and normalization
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        # [EDIT]: Initialize the AST Parser to clean code nodes during fetching
        self.ast_parser = ASTParser()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # 1. ROUTE: Buggy Image (Anchor)
        # Path: data/screenshots/spacing/Mut_01_SuccessAlert.png
        buggy_img_path = os.path.join(self.root_dir, "screenshots", self.category, entry['image_anchor'].split('/')[-1])
        buggy_img = Image.open(buggy_img_path).convert("RGB")

        # 2. ROUTE: Seed Image (Visual Baseline)
        # Extracts 'SuccessAlert' from ID to find the perfect reference screenshot
        component_name = entry['id'].split('_')[-1]
        seed_img_path = os.path.join(self.root_dir, "raw_seeds", "screenshots", self.category, f"{component_name}.png")
        seed_img = Image.open(seed_img_path).convert("RGB")

        # 3. ROUTE: Code Nodes (Positive & Negative)
        # [EDIT]: Passing raw strings through ASTParser to ensure clean tokenization
        pos_code = self.ast_parser.get_clean_node_text(entry['positive_node'])
        neg_code = self.ast_parser.get_clean_node_text(entry['negative_node'])

        # 4. ROUTE: Seed Code (Original Reference)
        # Loads the full healthy source code for syntactic context
        seed_code_path = os.path.join(self.root_dir, "raw_seeds", f"{self.category}.jsx")
        with open(seed_code_path, 'r') as f:
            raw_seed_content = f.read()
            # [EDIT]: Sanitize the full seed content
            full_seed_content = self.ast_parser.get_clean_node_text(raw_seed_content)

        # 5. TEXT: The user's natural language bug report
        text_query = entry['text_anchor']

        # Pre-processing images to 224x224 for the ViT Encoder
        pixel_values = self.processor(images=buggy_img, return_tensors="pt").pixel_values.squeeze(0)
        seed_pixels = self.processor(images=seed_img, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenizing text and sanitized code into 512-token context windows
        text_inputs = self.tokenizer(text_query, padding='max_length', truncation=True, max_length=512,
                                     return_tensors="pt")
        pos_inputs = self.tokenizer(pos_code, padding='max_length', truncation=True, max_length=512,
                                    return_tensors="pt")
        neg_inputs = self.tokenizer(neg_code, padding='max_length', truncation=True, max_length=512,
                                    return_tensors="pt")
        seed_inputs = self.tokenizer(full_seed_content, padding='max_length', truncation=True, max_length=512,
                                     return_tensors="pt")

        return {
            "anchor_img": pixel_values,
            "seed_img": seed_pixels,
            "anchor_text": text_inputs,
            "pos_code": pos_inputs,
            "neg_code": neg_inputs,
            "seed_code": seed_inputs
        }
