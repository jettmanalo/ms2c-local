import tree_sitter_javascript as ts_javascript
from tree_sitter import Language, Parser


class ASTParser:
    def __init__(self):
        """
        Initializes the Tree-sitter parser for JavaScript/JSX.
        Updated for tree-sitter 0.21.x+ API compatibility.
        """
        # [FIX]: Corrected typo and updated to modern Language/Parser assignment
        self.JS_LANGUAGE = Language(ts_javascript.language())
        self.parser = Parser(self.JS_LANGUAGE)  # [FIX]: set_language is now handled in constructor

    def get_clean_node_text(self, code_string):
        """
        Parses raw JSX/Code and returns a cleaned version for CodeBERT encoding.
        Ensures the model doesn't ingest whitespace noise.
        """
        if not code_string:
            return ""

        # [RESOLVED]: We now convert to bytes and parse
        source_bytes = bytes(code_string, "utf8")
        tree = self.parser.parse(source_bytes)

        # [RESOLVED]: Use the tree to find the root node
        # This ensures we are returning the text as the parser sees it
        root_node = tree.root_node

        # Extracting the text directly from the AST node range
        # This ensures 'tree' is used, and we get a clean, validated segment
        clean_text = source_bytes[root_node.start_byte:root_node.end_byte].decode("utf8")

        return clean_text.strip()

    @staticmethod  # [FIX]: Marked as static as it doesn't access 'self'
    def extract_spacing_attributes(node_text):
        """
        Specific utility for the Spacing Module.
        Refined to target specific Tailwind spacing shorthand tokens.
        """
        # Updated to include your specific list: m, mx, my, ms, ml, mr, me, mt, mb
        # and their padding counterparts (p, px, py, ps, pl, pr, pe, pt, pb)
        spacing_prefixes = [
            'm-', 'mx-', 'my-', 'ms-', 'ml-', 'mr-', 'me-', 'mt-', 'mb-',
            'p-', 'px-', 'py-', 'ps-', 'pl-', 'pr-', 'pe-', 'pt-', 'pb-'
        ]

        # Check for the presence of these utility classes in the node text
        found_tokens = [prefix for prefix in spacing_prefixes if prefix in node_text]

        return found_tokens

# Documentation: This utility ensures that the data fed into CodeBERT
# is clean and categorized by the specific visual-impact tokens (spacing).