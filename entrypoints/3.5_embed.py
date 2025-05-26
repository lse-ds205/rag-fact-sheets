import sys
import os
from pathlib import Path
import traceback
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm
import numpy as np

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.embedding import CombinedEmbedding
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.database import Connection, NDCDocumentORM as Document, DocChunkORM
from group4py.src.schema import DatabaseConfig
from group4py.src.hop_rag import HopRAGGraphProcessor

logger = logging.getLogger(__name__)

def collect_all_chunk_texts():
    """
    Collect all chunk texts from the database for global Word2Vec training.
    
    Returns:
        List of chunk texts
    """
    logger.info("[3.5_EMBED] Collecting all chunks from database for global Word2Vec training...")
    
    config = DatabaseConfig.from_env()
    connection = Connection(config)
    connection.connect()
    session = connection.get_session()
    
    try:
        # Get all chunks with their content
        chunks = session.query(DocChunkORM.content).filter(DocChunkORM.content.isnot(None)).all()
        texts = [chunk.content for chunk in chunks if chunk.content and chunk.content.strip()]
        
        logger.info(f"[3.5_EMBED] Collected {len(texts)} chunks from database")
        return texts
        
    except Exception as e:
        logger.error(f"[3.5_EMBED] Error collecting chunks: {e}")
        return []
    finally:
        session.close()


async def embed_all_chunks(force_reembed: bool = False):
    """
    Generate embeddings for all chunks in the database.
    
    Args:
        force_reembed: If True, regenerate embeddings even if they already exist
    """
    logger.info("[3.5_EMBED] Starting embedding generation process...")
    
    # Step 1: Collect all chunk texts and train global Word2Vec
    chunk_texts = collect_all_chunk_texts()
    
    if not chunk_texts:
        logger.error("[3.5_EMBED] No chunks found in database")
        return
    
    # Step 2: Initialize embedding models
    logger.info("[3.5_EMBED] Loading embedding models...")
    embedding_model = CombinedEmbedding()
    
    # Train or load Word2Vec model
    model_path = project_root / "local_models" / "word2vec"
    
    # Check if model already exists
    if model_path.exists() and not force_reembed:
        logger.info(f"[3.5_EMBED] Loading existing Word2Vec model from {model_path}")
        embedding_model.word2vec_embedder.load_global_model(str(model_path))
    else:
        logger.info("[3.5_EMBED] Training new global Word2Vec model...")
        success = embedding_model.train_word2vec_on_texts(chunk_texts, str(model_path))
        if not success:
            logger.error("[3.5_EMBED] Failed to train Word2Vec model")
            return
    
    # Load transformer models
    embedding_model.transformer_embedder.load_models()
    
    if not embedding_model.models_ready:
        logger.error("[3.5_EMBED] No embedding models loaded successfully")
        return

    # Step 3: Get all chunks from database
    config = DatabaseConfig.from_env()
    connection = Connection(config)
    connection.connect()
    session = connection.get_session()
    
    try:
        # Query chunks that need embeddings (or all if force_reembed)
        if force_reembed:
            chunks_query = session.query(DocChunkORM).all()
            logger.info(f"[3.5_EMBED] Force re-embedding: Processing all {len(chunks_query)} chunks")
        else:
            chunks_query = session.query(DocChunkORM).filter(
                (DocChunkORM.transformer_embedding.is_(None)) | 
                (DocChunkORM.word2vec_embedding.is_(None))
            ).all()
            logger.info(f"[3.5_EMBED] Processing {len(chunks_query)} chunks without embeddings")
        
        if not chunks_query:
            logger.info("[3.5_EMBED] No chunks need embedding. All done!")
            return
        
        # Step 4: Generate embeddings for all chunks
        logger.info("[3.5_EMBED] Generating embeddings for chunks...")
        
        processed_count = 0
        for chunk in tqdm(chunks_query, desc="Generating embeddings"):
            try:
                if not chunk.content or not chunk.content.strip():
                    logger.warning(f"[3.5_EMBED] Skipping empty chunk {chunk.id}")
                    continue
                
                # Generate transformer embedding
                transformer_embedding = embedding_model.transformer_embedder.embed_transformer(chunk.content)
                
                # Generate Word2Vec embedding using global model
                word2vec_embedding = embedding_model.word2vec_embedder.embed_text(chunk.content)
                
                # Update chunk with embeddings
                if transformer_embedding and len(transformer_embedding) > 0:
                    # Ensure all elements are floats
                    transformer_embedding = [float(val) if not isinstance(val, float) else val for val in transformer_embedding]
                    chunk.transformer_embedding = transformer_embedding
                
                if word2vec_embedding is not None and len(word2vec_embedding) > 0:
                    # Ensure all elements are floats and convert to list
                    word2vec_embedding = [float(val) if not isinstance(val, float) else val for val in word2vec_embedding.tolist()]
                    chunk.word2vec_embedding = word2vec_embedding
                
                processed_count += 1
                
                # Commit periodically to avoid holding large transactions (every 500 chunks)
                if processed_count % 500 == 0:
                    session.commit()
                    logger.debug(f"[3.5_EMBED] Committed batch at {processed_count} chunks")
                    
            except Exception as e:
                logger.error(f"[3.5_EMBED] Error processing chunk {str(chunk.id)}: {str(e)}")
                logger.error(f"[3.5_EMBED] Traceback: {traceback.format_exc()}")
                continue
        
        # Final commit
        session.commit()
        logger.info(f"[3.5_EMBED] Successfully generated embeddings for {processed_count} chunks")
        
        # Step 5: Run HopRAG processing after embeddings are complete
        try:
            logger.info("[3.5_EMBED] Starting HopRAG processing...")
            config = DatabaseConfig.from_env()
            processor = HopRAGGraphProcessor(config)
            await processor.initialize()
            
            # Generate embeddings if needed (HopRAG uses its own embedding format)
            logger.info("[3.5_EMBED] Processing HopRAG embeddings in batch...")
            await processor.process_embeddings_batch(batch_size=100)
            
            # Build relationships for all chunks with enhanced logging
            logger.info("[3.5_EMBED] Building logical relationships...")
            
            # Get the current relationship count before building
            rel_count_before = 0
            try:
                with processor.db_connection.get_engine().connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
                    rel_count_before = result.scalar() or 0
                    logger.info(f"[3.5_EMBED] Current relationship count: {rel_count_before}")
            except Exception as e:
                logger.warning(f"[3.5_EMBED] Could not get relationship count: {str(e)}")
            
            # Build relationships with detailed parameters
            logger.info("[3.5_EMBED] Building relationships for all processed chunks...")
            await processor.build_relationships_sparse(
                max_neighbors=30, 
                min_confidence=0.55,
                force_commit=True  # Ensure relationships are committed to database
            )
            
            # Check how many relationships were added
            try:
                with processor.db_connection.get_engine().connect() as conn:
                    # Get total relationships
                    result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
                    rel_count_after = result.scalar() or 0
                    
                    new_rels = rel_count_after - rel_count_before
                    logger.info(f"[3.5_EMBED] Added {new_rels} new relationships. Total now: {rel_count_after}")
            except Exception as e:
                logger.warning(f"[3.5_EMBED] Could not get updated relationship count: {str(e)}")
            
            # Clean up HopRAG processor
            await processor.close()
            logger.info("[3.5_EMBED] HopRAG processing completed successfully")
            
            # Ensure any remaining changes from HopRAG are committed to the database
            session.commit()
            logger.info("[3.5_EMBED] Successfully committed HopRAG relationship changes")
            
        except Exception as e:
            session.rollback()  # Roll back any failed HopRAG changes
            logger.error(f"[3.5_EMBED] HopRAG processing failed: {str(e)}")
            logger.error(f"[3.5_EMBED] HopRAG Traceback: {traceback.format_exc()}")
            # Don't fail the entire process if HopRAG fails - embeddings are still valid
        
    except Exception as e:
        logger.error(f"[3.5_EMBED] Error during embedding generation: {str(e)}")
        session.rollback()
        raise
    finally:
        session.close()


@Logger.log(log_file = project_root / "logs/embed.log", log_level="INFO")
async def run_script(force_reembed: bool = False):
    """
    Main function to generate embeddings for all chunks.
    
    Args:
        force_reembed: If True, regenerate embeddings even if they already exist
    """
    try:
        logger.warning(f"\n\n[3.5_EMBED] Running embedding script with force_reembed={force_reembed}...")
        
        # Generate embeddings for all chunks
        await embed_all_chunks(force_reembed=force_reembed)
        
        logger.warning("[3.5_EMBED] Embedding and relationship processing completed successfully. All chunks now have embeddings and logical relationships.")
        
    except Exception as e:
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 3.5_embed.py: {e}")
        logger.critical(f"[PIPELINE BROKE!] - Traceback: {traceback.format_exc()}")
        raise e


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Generate embeddings for document chunks using global Word2Vec and transformers")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of embeddings, even if they already exist")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the script with force_reembed parameter
    asyncio.run(run_script(force_reembed=args.force))