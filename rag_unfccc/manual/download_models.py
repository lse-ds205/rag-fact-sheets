# One-off script to download models - not included in continuous pipeline
import os
import logging
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths for models
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'local_models')
ENGLISH_MODEL_DIR = os.path.join(MODELS_DIR, 'distilroberta-base')
MULTILINGUAL_MODEL_DIR = os.path.join(MODELS_DIR, 'xlm-roberta-base')

def download_models():
    """
    Directs downloaded models to a .gitignore'd directory
    """
    # Create directories if they don't exist
    os.makedirs(ENGLISH_MODEL_DIR, exist_ok=True)
    os.makedirs(MULTILINGUAL_MODEL_DIR, exist_ok=True)
    
    try:
        # Download English model from Hugging Face
        model_name = "climatebert/distilroberta-base-climate-f"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=ENGLISH_MODEL_DIR)
        model = AutoModel.from_pretrained(model_name, cache_dir=ENGLISH_MODEL_DIR)
        
        logger.info(f"Successfully downloaded English model to {ENGLISH_MODEL_DIR}")

        # Download multilingual model from Hugging Face
        model_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MULTILINGUAL_MODEL_DIR)
        model = AutoModel.from_pretrained(model_name, cache_dir=MULTILINGUAL_MODEL_DIR)
        
        logger.info(f"Successfully downloaded multilingual model to {MULTILINGUAL_MODEL_DIR}")
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        raise

if __name__ == "__main__":
    download_models()