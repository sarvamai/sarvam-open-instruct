"""
Example of conversation generation using Gemini through Curator.
This module provides functionality to generate multi-turn conversations in different languages.
"""

import os
import logging
from typing import Dict, Any, List
from bespokelabs import curator
from datasets import load_dataset, Dataset
import random
from tqdm import tqdm

# Configure logging
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)

# Language configurations with English added and samples specified
LANGUAGES = [
    {"lang": "English", "lang_short": "en", "samples": 100},
    {"lang": "Hindi", "lang_short": "hi", "samples": 100},
    {"lang": "Bengali", "lang_short": "bn", "samples": 100},
    {"lang": "Telugu", "lang_short": "te", "samples": 100},
    {"lang": "Kannada", "lang_short": "kn", "samples": 100},
    {"lang": "Marathi", "lang_short": "mr", "samples": 100},
    {"lang": "Tamil", "lang_short": "ta", "samples": 100},
    {"lang": "Gujarati", "lang_short": "gu", "samples": 100},
    {"lang": "Punjabi", "lang_short": "pa", "samples": 100},
    {"lang": "Odia", "lang_short": "or", "samples": 100},
    {"lang": "Malayalam", "lang_short": "ml", "samples": 100},
    
    
]

# Default generation parameters for Gemini
DEFAULT_GENERATION_PARAMS = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

class ConversationGenerator(curator.LLM):
    """A conversation generator using Gemini model."""
    
    def __init__(
        self,
        model_name: str = "gemini-2-flash-001",
        backend: str = "gemini",
        generation_params: Dict[str, Any] | None = None,
    ):
        """Initialize the generator."""
        super().__init__(
            model_name=model_name,
            backend=backend,
            batch=True,
            generation_params=generation_params or DEFAULT_GENERATION_PARAMS
        )
        self.prompt_func = self.prompt

    def prompt(self, input: Dict[str, Any]) -> str:
        """Generate the conversation prompt."""
        lang_config = input["lang_config"]
        #task_name = input["task_name"]
        prompt = input["prompt"]
        prompt = f"""

Given the persona from the Anthropic Index: {prompt}, please generate three example prompts that this persona might ask. The prompts should vary in difficulty: easy, medium, and hard and in the language {lang_config}. For each prompt, also provide a category in English.

Please format your response in the following JSON structure:


{{
  "easy": "Prompt text in {lang_config}",
  "category_easy": "Category in English",
  "medium": "Prompt text in {lang_config}",
  "category_medium": "Category in English",
  "hard": "Prompt text in {lang_config}",
  "category_hard": "Category in English"
}}
Ensure that the prompts are relevant to the persona and reflect increasing levels of complexity or specificity.
- This is for a text LLM, so prompt should be related to text related tasks.
- Don't add reference to the persona.
- Make sure each prompt is multi-line, for all difficulty levels. single line prompts are not allowed.
"""
        return prompt

    def parse(self, input: Dict[str, Any], response: str) -> Dict[str, Any]:
        """Parse the model's response."""
        return {
            "task_name": input["prompt"],
            "language": input["lang_config"]["lang"],
            "lang_code": input["lang_config"]["lang_short"],
            "result": response.strip()
        }

def prepare_generation_inputs(dataset: Dataset, languages: List[Dict]) -> List[Dict]:
    """Prepare inputs for conversation generation."""
    generation_inputs = []
    
    for lang_config in languages:
        # Get number of samples for this language
        samples = lang_config["samples"]
        
        # Sample random indices for this language
        total_rows = len(dataset)
        #fix the next line to get the same samples for each language

        indices = random.sample(range(total_rows),samples)
        
        # Create inputs for this language
        for idx in indices:
            generation_inputs.append({
                "prompt": dataset[idx]["task_name"],
                "lang_config": lang_config,
                "row_idx": idx
            })
    
    logger.info(f"Prepared {len(generation_inputs)} generation inputs")
    return generation_inputs

def main():
    """Example usage of the ConversationGenerator."""
    try:
        # Load dataset
        logger.info("Loading dataset...")
        ds = load_dataset("Anthropic/EconomicIndex")
        dataset = ds['train']
        
        # Prepare generation inputs
        logger.info("Preparing generation inputs...")
        generation_inputs = prepare_generation_inputs(dataset, LANGUAGES)
        
        # Configure generator
        generator = ConversationGenerator(
            model_name="gemini-1.5-pro-002",
            backend="gemini",
            generation_params=DEFAULT_GENERATION_PARAMS
        )
        
        # Generate conversations
        logger.info("Generating conversations...")
        conversations = generator(generation_inputs)
        
        # Convert to dataset
        conversation_ds = Dataset.from_list(conversations)
        
        # Save the conversations
        output_dir = "conversations"
        os.makedirs(output_dir, exist_ok=True)
        conversation_ds.save_to_disk(os.path.join(output_dir, "conversations"))
        
        logger.info("Conversation generation completed successfully!")
        logger.info(f"Results saved to {output_dir}/indic_persona_prompts/")
        logger.info(f"Dataset structure: {conversation_ds.features}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error("Conversation data structure:")
        if 'conversations' in locals():
            for c in conversations[:2]:  # Show first two conversations
                logger.error(f"Conversation entry: {c}")
        raise

if __name__ == "__main__":
    main()