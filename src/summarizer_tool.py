from smolagents import Tool
from typing import Dict, Any, Optional
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

class TextSummarizerTool(Tool):
    name = "text_summarizer"
    description = """
    Summarizes text using various summarization methods and models.
    This tool can generate concise summaries of longer texts while preserving key information.
    It supports different summarization models and customizable parameters.
    """
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to be summarized",
        },
        "model": {
            "type": "string",
            "description": "Summarization model to use (default: 'facebook/bart-large-cnn')",
            "nullable": True
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of the summary in tokens (default: 130)",
            "nullable": True
        },
        "min_length": {
            "type": "integer",
            "description": "Minimum length of the summary in tokens (default: 30)",
            "nullable": True
        },
        "style": {
            "type": "string",
            "description": "Style of summary: 'concise', 'detailed', or 'bullet_points' (default: 'concise')",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self):
        """Initialize the Text Summarizer Tool with default settings."""
        super().__init__()
        self.default_model = "facebook/bart-large-cnn"
        self.available_models = {
            "facebook/bart-large-cnn": "BART CNN (good for news)",
            "sshleifer/distilbart-cnn-12-6": "DistilBART (faster, smaller)",
            "google/pegasus-xsum": "Pegasus (extreme summarization)",
            "facebook/bart-large-xsum": "BART XSum (very concise)",
            "philschmid/bart-large-cnn-samsum": "BART SamSum (good for conversations)"
        }
        # Pipeline will be lazily loaded
        self._pipeline = None
        
    def _load_pipeline(self, model_name: str):
        """Load the summarization pipeline with the specified model."""
        try:
            from transformers import pipeline
            import torch
            
            # Try to detect if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Load the summarization pipeline
            self._pipeline = pipeline(
                "summarization",
                model=model_name,
                device=device
            )
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            try:
                # Fall back to default model
                from transformers import pipeline
                import torch
                device = 0 if torch.cuda.is_available() else -1
                self._pipeline = pipeline(
                    "summarization",
                    model=self.default_model,
                    device=device
                )
                return True
            except Exception as fallback_error:
                print(f"Error loading fallback model: {str(fallback_error)}")
                return False
                
    def _format_as_bullets(self, summary: str) -> str:
        """Format a summary as bullet points."""
        # Split the summary into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Format as bullet points
        bullet_points = []
        for sentence in sentences:
            # Skip very short sentences that might be artifacts
            if len(sentence) < 15:
                continue
            bullet_points.append(f"• {sentence}")
            
        return "\n".join(bullet_points)
                
    def forward(self, text: str, model: str = None, max_length: int = None, min_length: int = None, style: str = None) -> str:
        """
        Summarize the input text.
        
        Args:
            text: The text to summarize
            model: Summarization model to use
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            style: Style of summary ('concise', 'detailed', or 'bullet_points')
            
        Returns:
            Summarized text
        """
        # Set default values if parameters are None
        if model is None:
            model = self.default_model
        if max_length is None:
            max_length = 130
        if min_length is None:
            min_length = 30
        if style is None:
            style = "concise"
            
        # Validate model choice
        if model not in self.available_models:
            return f"Model '{model}' not recognized. Available models: {', '.join(self.available_models.keys())}"
            
        # Load the model if not already loaded or if different from current
        if self._pipeline is None or (hasattr(self._pipeline, 'model') and self._pipeline.model.name_or_path != model):
            if not self._load_pipeline(model):
                return "Failed to load summarization model. Please try a different model."
                
        # Adjust parameters based on style
        if style == "concise":
            max_length = min(100, max_length)
            min_length = min(30, min_length)
        elif style == "detailed":
            max_length = max(150, max_length)
            min_length = max(50, min_length)
            
        # Ensure text is not too short
        if len(text.split()) < 20:
            return "The input text is too short to summarize effectively."
                
        # Perform summarization
        try:
            # Truncate very long inputs if needed (model dependent)
            max_input_length = 1024  # Most models have limits around 1024-2048 tokens
            words = text.split()
            if len(words) > max_input_length:
                text = " ".join(words[:max_input_length])
                note = "\n\nNote: The input was truncated due to length limits."
            else:
                note = ""
                
            summary = self._pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            result = summary[0]['summary_text']
            
            # Format the result based on style
            if style == "bullet_points":
                result = self._format_as_bullets(result)
                
            # Add metadata
            metadata = f"\n\nSummarized using: {self.available_models.get(model, model)}"
            
            return result + metadata + note
            
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
            
    def get_available_models(self) -> Dict[str, str]:
        """Return the dictionary of available models with descriptions."""
        return self.available_models

# Example usage:
# summarizer = TextSummarizerTool()
# result = summarizer("Long text goes here...", model="facebook/bart-large-cnn", style="bullet_points")
# print(result)