
import logging
import os
import glob
import uuid
import torch
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm, trange

from transformers import AutoTokenizer, AutoModel

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sqlalchemy.orm import sessionmaker
from functions import generate_embeddings_for_text  # Assuming this is defined elsewhere
import os
import re
from gensim.models import Word2Vec
from tqdm.notebook import tqdm as tqdm


# Set up logger
logger = logging.getLogger(__name__)

# Suppress transformers warnings
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")



def generate_embeddings_for_text(texts, model, tokenizer):
    """
    Get embeddings from a pre-trained transformer model.
    
    Args:
        texts: List of input texts
        model: Transformer model
        tokenizer: Tokenizer for the model
        
    Returns:
        NumPy array of embeddings for the input texts
    """

    # Tokenize the input texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Get model output (without gradient calculation for efficiency)
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Use the CLS token embedding as the sentence embedding
    # This is a simple approach - in practice, you might use more sophisticated pooling
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    
    return sentence_embeddings.numpy().flatten().tolist()


def train_custom_word2vec_from_texts(
    texts,
    vector_size=768,
    window=5,
    min_count=1,
    workers=4,
    epochs=10,
    save_path="./local_model/custom_word2vec_768.model",
    force_include_words=None
):
    """
    Trains a Word2Vec model from raw texts and saves it, ensuring rare but important words are included.
    
    Args:
        texts (List[str]): List of raw document texts.
        vector_size (int): Desired dimensionality.
        window (int): Context window size.
        min_count (int): Minimum word frequency.
        workers (int): CPU cores to use.
        epochs (int): Training epochs.
        save_path (str): Where to save the trained model.
        force_include_words (List[str]): Words to artificially include in training.
    
    Returns:
        model (Word2Vec): Trained Word2Vec model.
    """

    def basic_tokenize(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
        return text.split()

    tokenized_docs = [basic_tokenize(doc) for doc in texts if isinstance(doc, str) and doc.strip()]

    # ⏫ Add synthetic sentences to ensure inclusion of rare/important terms
    if force_include_words:
        booster_sentences = [[word] for word in force_include_words]
        tokenized_docs.extend(booster_sentences)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )

    model.save(save_path)
    return model



def load_climatebert_model():
    """
    Load the ClimateBERT model and tokenizer from HuggingFace.
    
    Returns:
        model: The pre-trained ClimateBERT model.
        tokenizer: The tokenizer for the model.
    """
    EMBEDDING_MODEL_LOCAL_DIR = os.getenv("EMBEDDING_MODEL_LOCAL_DIR")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_LOCAL_DIR)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_LOCAL_DIR)
    return tokenizer, model




def generate_word2vec_embedding_for_text(text, model):
    tokens = simple_preprocess(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0).tolist()
    else:
        return [0.0] * model.vector_size




def embed_and_store_all_embeddings(df, engine):


    # Load models
    tokenizer, climatebert_model = load_climatebert_model()
    word2vec_model = Word2Vec.load("./local_model/custom_word2vec_768.model")

    # Ensure geographies are strings
    df["document_metadata.geographies"] = df["document_metadata.geographies"].astype(str)

    # Extract 3-letter country codes from curly braces
    df["country_code"] = df["document_metadata.geographies"].str.extract(r"\{(\w+)\}")
    country_codes = df["country_code"].dropna().unique()

    # Group documents by country
    country_chunks = {}
    for code in tqdm(country_codes, desc="Filtering by country"):
        country_chunks[code] = df[df["country_code"] == code]

    # Set batch size
    batch_size = 10000

    # Prepare DB session
    Session = sessionmaker(bind=engine)
    session = Session()

    for code, chunk in tqdm(country_chunks.items(), desc="Processing all countries"):
        texts = chunk["text_block.text"]
        doc_ids = chunk["document_id"]
        source_urls = chunk["document_metadata.source_url"]
        titles = chunk["document_metadata.document_title"]

        num_batches = (len(texts) + batch_size - 1) // batch_size

        all_doc_ids = []
        all_texts = []
        all_urls = []
        all_titles = []
        all_climatebert_embeddings = []
        all_word2vec_embeddings = []

        for i in tqdm(range(num_batches), desc=f"Embedding {code}", leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))

            batch_texts = texts.iloc[start_idx:end_idx].reset_index(drop=True)
            batch_ids = doc_ids.iloc[start_idx:end_idx].reset_index(drop=True)
            batch_urls = source_urls.iloc[start_idx:end_idx].reset_index(drop=True)
            batch_titles = titles.iloc[start_idx:end_idx].reset_index(drop=True)

            mask = batch_texts.apply(lambda x: isinstance(x, str) and x.strip() != "")
            batch_texts = batch_texts[mask]
            batch_ids = batch_ids[mask]
            batch_urls = batch_urls[mask]
            batch_titles = batch_titles[mask]

            batch_climatebert_embeddings = batch_texts.progress_apply(
                lambda text: generate_embeddings_for_text(text, climatebert_model, tokenizer)
            )
            batch_word2vec_embeddings = batch_texts.progress_apply(
                lambda text: generate_word2vec_embedding_for_text(text, word2vec_model)
            )

            all_doc_ids.extend(batch_ids)
            all_texts.extend(batch_texts)
            all_urls.extend(batch_urls)
            all_titles.extend(batch_titles)
            all_climatebert_embeddings.extend(batch_climatebert_embeddings)
            all_word2vec_embeddings.extend(batch_word2vec_embeddings)

        for doc_id, cbert_emb, w2v_emb, original_text, url, title in tqdm(
            zip(all_doc_ids, all_climatebert_embeddings, all_word2vec_embeddings, all_texts, all_urls, all_titles),
            total=len(all_doc_ids),
            desc=f"Uploading {code}"
        ):
            stmt = text("""
                INSERT INTO document_embeddings 
                    (document_id, document_title, country_code, original_text, source_hyperlink, climatebert_embedding, word2vec_embedding)
                VALUES 
                    (:document_id, :document_title, :country_code, :original_text, :source_hyperlink, :climatebert_embedding, :word2vec_embedding)
            """)

            session.execute(stmt, {
                "document_id": doc_id,
                "document_title": title,
                "country_code": code,
                "original_text": original_text,
                "source_hyperlink": url,
                "climatebert_embedding": cbert_emb,
                "word2vec_embedding": w2v_emb
            })

        session.commit()

    print("\n✅ All ClimateBERT and Word2Vec embeddings uploaded directly.")


def store_database_batched(flat_ds, num_chunks, batch_size=100000):
    """
    Store the flattened dataset into a PostgreSQL database using SQLAlchemy in chunks with progress bar.

    """
    load_dotenv()
    engine = create_engine(os.getenv("DB_URL"))
    df = pd.DataFrame(flat_ds[:1])

    # Initialize table with first chunk
    df.to_sql('climate_policy_radar', engine, if_exists='replace', index=False)

    for i in tqdm(range(1, num_chunks, batch_size), desc="Inserting chunks into database"):
        chunk = pd.DataFrame(flat_ds[i:i+batch_size])
        chunk.to_sql('climate_policy_radar', engine, if_exists='append', index=False)