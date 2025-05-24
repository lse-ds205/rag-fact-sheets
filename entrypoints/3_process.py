import sys
from pathlib import Path
import traceback
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dotenv import load_dotenv
import os 
from sqlalchemy import create_engine, text
from datetime import datetime, date
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import uuid
from uuid import UUID

# Load environment variables from .env file
load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.extract_document import extract_text_from_pdf
from group4py.src.chunk_embed import DocChunk as DocChunker, Embedding
from group4py.src.helpers import Logger, Test, TaskInfo
from group4py.src.schema import DocChunk
from group4py.src.database import Connection, Document, DocChunkORM
from group4py.src.constants.settings import FILE_PROCESSING_CONCURRENCY
from group4py.src.hop_rag import HopRAGGraphProcessor
from group4py.src.schema import DatabaseConfig

logger = logging.getLogger(__name__)

def get_file_paths():
    """
    Get the file paths of PDF files in the data/pdfs folder.
    """
    pdf_directory = project_root / "data" / "pdfs"
    if not pdf_directory.exists():
        logger.warning(f"[3_PROCESS] PDF directory not found: {pdf_directory}")
        return []
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    logger.info(f"[3_PROCESS] Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    # Convert Path objects to strings
    return [str(file_path) for file_path in pdf_files]

async def process_file_one(file_path: str, force_reprocess: bool = False):
    """
    Process a file and return a list of chunks and embeddings.
    
    Args:
        file_path: Path to the PDF file
        force_reprocess: If True, reprocess the file even if it has been processed before
    """
    try:
        # Extract file name to be used as document ID
        file_name = Path(file_path).stem
        logger.info(f"[3_PROCESS] Processing file {file_path}")
        
        # Check if document has already been processed
        connection = Connection()
        doc_id = file_name
        
        # Use the new check_document_processed function
        is_processed, document = connection.check_document_processed(doc_id)
        
        if is_processed and not force_reprocess:
            logger.info(f"[3_PROCESS] Document {doc_id} has already been processed. Skipping.")
            return None
            
        # If force_reprocess is True and document exists, we need to delete existing chunks
        if force_reprocess and document:
            logger.info(f"[3_PROCESS] Force reprocessing document {doc_id}")
            engine = connection.get_engine()
            session = connection.get_session()
            try:                # Remove existing chunks for this document
                session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_id).delete()
                session.commit()
                logger.info(f"[3_PROCESS] Removed existing chunks for document {doc_id}")
            except Exception as e:
                logger.error(f"[3_PROCESS] Error removing existing chunks: {e}")
                session.rollback()
            finally:
                session.close()
        
        # If document doesn't exist, create it first before proceeding with chunk processing
        if not document:
            logger.info(f"[3_PROCESS] Document {doc_id} not found in database. Creating record...")
            session = connection.get_session()
            
            try:
                # Extract document metadata from filename
                parts = doc_id.split('_')
                country = parts[0].replace('_', ' ').title() if len(parts) > 0 else ''
                language = parts[1] if len(parts) > 1 else ''
                
                # Try to parse date if available (format: YYYYMMDD)
                submission_date = None
                if len(parts) > 2 and len(parts[2]) >= 8:
                    date_str = parts[2][:8]  # Extract YYYYMMDD portion
                    try:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        submission_date = date(year, month, day)
                    except (ValueError, IndexError):
                        logger.warning(f"[3_PROCESS] Could not parse date from filename: {parts[2]}")
                
                # Create document record
                new_document = Document(
                    doc_id=doc_id,
                    country=country,
                    title=f"{country} NDC",
                    url="",  # Empty URL since we're processing local files
                    language=language,
                    submission_date=submission_date,
                    file_path=file_path,
                    file_size=Path(file_path).stat().st_size / (1024 * 1024)  # Size in MB
                )
                
                session.add(new_document)
                session.commit()
                logger.info(f"[3_PROCESS] Created document record for {doc_id}")
                document = new_document
            except Exception as e:
                logger.error(f"[3_PROCESS] Error creating document record: {e}")
                session.rollback()
            finally:
                session.close()
        
        # 1. Extract text from PDF
        extracted_elements = extract_text_from_pdf(file_path)
        if not extracted_elements:
            logger.error(f"[3_PROCESS] No text extracted from {file_path}")
            return None
        
        logger.info(f"[3_PROCESS] Extracted {len(extracted_elements)} elements from {file_path}")
        
        # 2. Create chunks from the extracted text
        doc_chunker = DocChunker()
        chunks = doc_chunker.chunk_document_by_sentences(extracted_elements)
        logger.info(f"[3_PROCESS] Created {len(chunks)} chunks from {file_path}")
        
        # 3. Clean chunks
        try:
            cleaned_chunks = doc_chunker.cleaning_function(chunks)
            logger.info(f"[3_PROCESS] Finished cleaning chunks from {file_path}, got {len(cleaned_chunks)} cleaned chunks")
        except Exception as e:
            logger.error(f"[3_PROCESS] Error during chunk cleaning: {e}")
            # Fall back to using the original chunks if cleaning fails
            logger.info(f"[3_PROCESS] Falling back to original chunks")
            cleaned_chunks = chunks
        
        # Validate cleaned chunks before proceeding
        if not cleaned_chunks or len(cleaned_chunks) == 0:
            logger.error(f"[3_PROCESS] No valid chunks after cleaning for {file_path}")
            return None
        # 4. Create DocChunk objects before uploading
        db_chunks = []
        for i, chunk in enumerate(cleaned_chunks):
            # Skip empty chunks
            if not chunk.get('text', '').strip():
                logger.warning(f"[3_PROCESS] Skipping empty chunk {i} from {file_path}")
                continue
                  # Create a DocChunkORM instance
            chunk_model = DocChunkORM(
                id=str(uuid.uuid4()),
                doc_id=file_name,
                content=chunk.get('text', ''),
                chunk_index=i,
                paragraph=chunk.get('metadata', {}).get('paragraph_number'),
                language=chunk.get('metadata', {}).get('language'),
                chunk_data=chunk.get('metadata', {})
            )
            
            db_chunks.append(chunk_model)
        
        # Verify we have chunks to upload
        if not db_chunks:
            logger.error(f"[3_PROCESS] No valid chunks to upload for {file_path}")
            return None
            
        # 5. Upload chunks to the database, specifically to doc_chunks table
        logger.info(f"[3_PROCESS] Attempting to upload {len(db_chunks)} chunks to database for {file_path}")
        upload_success = connection.upload(db_chunks, table='doc_chunks')
        if not upload_success:
            logger.error(f"[3_PROCESS] Failed to upload chunks to database for {file_path}")
            return None
            
        logger.info(f"[3_PROCESS] Uploaded {len(db_chunks)} chunks to doc_chunks table")

        # 6. Update documents table that the document has been processed
        connection.update_processed(Document, doc_id, chunks=cleaned_chunks, table='documents')
        logger.info(f"[3_PROCESS] Updated processed status in documents table for {doc_id}")
        
        # 7. Create transformer embeddings
        try:
            embedder = Embedding()
            # Explicitly load the models before embedding
            embedder.load_models()
            
            # Check if embedding models were loaded successfully
            if embedder.models_loaded:
                logger.info(f"[3_PROCESS] Embedding models loaded successfully, generating embeddings for {len(cleaned_chunks)} chunks")
                transformer_embedded_chunks = embedder.embed_many(cleaned_chunks)
                logger.info(f"[3_PROCESS] Created transformer embeddings for {len(transformer_embedded_chunks)} chunks")
            else:
                logger.error(f"[3_PROCESS] Embedding models failed to load, falling back to chunks without embeddings")
                transformer_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        except Exception as e:
            logger.error(f"[3_PROCESS] Error creating transformer embeddings: {e}")
            transformer_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        
        # 8. Create word2vec embeddings
        try:
            # Check if we have valid transformer embeddings before attempting word2vec
            if embedder.models_loaded:
                embedder_word2vec = Embedding()
                word2vec_embedded_chunks = embedder_word2vec.word2vec_embedding(cleaned_chunks)
                logger.info(f"[3_PROCESS] Created word2vec embeddings for {len(word2vec_embedded_chunks)} chunks")
            else:
                logger.warning(f"[3_PROCESS] Skipping word2vec embeddings since transformer models failed to load")
                word2vec_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        except Exception as e:
            logger.error(f"[3_PROCESS] Error creating word2vec embeddings: {e}")
            word2vec_embedded_chunks = cleaned_chunks  # Use original chunks without embeddings
        
        # 9. Process the embeddings and update the database
        session = connection.get_session()

        try:            
            # Retrieve the existing chunks from the database to get their IDs
            db_chunk_query = session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_id).all()
            db_chunk_dict = {chunk.chunk_index: chunk for chunk in db_chunk_query}
            
            # Update each chunk with its embeddings
            for i, chunk in enumerate(cleaned_chunks):
                # Skip chunks that don't exist in the database
                if i not in db_chunk_dict:
                    logger.warning(f"[3_PROCESS] Chunk with index {i} for document {doc_id} not found in database")
                    continue
                    
                # Get the transformer embedding for this chunk
                transformer_embedding = None
                if i < len(transformer_embedded_chunks) and isinstance(transformer_embedded_chunks[i], dict):
                    transformer_embedding = transformer_embedded_chunks[i].get('embedding', [])
                    
                # Get the word2vec embedding for this chunk
                word2vec_embedding = None
                if i < len(word2vec_embedded_chunks) and isinstance(word2vec_embedded_chunks[i], dict):
                    word2vec_embedding = word2vec_embedded_chunks[i].get('w2v_embedding', [])
                
                # Get the corresponding database object
                db_chunk = db_chunk_dict[i]
                
                # Update embeddings if they exist and are valid
                if transformer_embedding and isinstance(transformer_embedding, list) and len(transformer_embedding) > 0:
                    # Ensure all elements are floats
                    transformer_embedding = [float(val) if not isinstance(val, float) else val for val in transformer_embedding]
                    db_chunk.transformer_embedding = transformer_embedding
                    logger.debug(f"[3_PROCESS] Updated transformer embedding for chunk {i} (length: {len(transformer_embedding)})")
                
                if word2vec_embedding and isinstance(word2vec_embedding, list) and len(word2vec_embedding) > 0:
                    # Ensure all elements are floats
                    word2vec_embedding = [float(val) if not isinstance(val, float) else val for val in word2vec_embedding]
                    db_chunk.word2vec_embedding = word2vec_embedding
                    logger.debug(f"[3_PROCESS] Updated word2vec embedding for chunk {i} (length: {len(word2vec_embedding)})")
        
            # Commit embedding changes
            session.commit()
            logger.info(f"[3_PROCESS] Successfully updated embeddings for document {doc_id}")
            
            # 10. Run HopRAG processing after embeddings are complete
            try:
                logger.info(f"[3_PROCESS] Running HopRAG processing for {doc_id}")
                config = DatabaseConfig.from_env()
                processor = HopRAGGraphProcessor(config)
                await processor.initialize()
                
                # Generate embeddings if needed
                logger.info(f"[3_PROCESS] Processing embeddings in batch for {doc_id}")
                await processor.process_embeddings_batch(batch_size=100)
                
                # Build relationships for this document's chunks with enhanced logging
                logger.info(f"[3_PROCESS] Building logical relationships for {doc_id}")
                
                # Get the current relationship count before building
                rel_count_before = 0
                try:
                    with processor.db_connection.get_engine().connect() as conn:
                        result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
                        rel_count_before = result.scalar() or 0
                        logger.info(f"[3_PROCESS] Current relationship count: {rel_count_before}")
                except Exception as e:
                    logger.warning(f"[3_PROCESS] Could not get relationship count: {e}")
                
                # Build relationships with more detailed parameters
                # Add filtering to focus only on chunks from the current document
                logger.info(f"[3_PROCESS] Building relationships specifically for document: {doc_id}")
                doc_chunk_count = await processor.get_doc_chunk_count(doc_id)
                logger.info(f"[3_PROCESS] Document {doc_id} has {doc_chunk_count} chunks to process for relationships")
                
                # Ensure we're focused on this document's chunks
                await processor.build_relationships_sparse(
                    max_neighbors=30, 
                    min_confidence=0.55,
                    doc_id=doc_id,  # Add document ID filter to focus on current document
                    force_commit=True  # Ensure relationships are committed to database
                )
                
                # Check how many relationships were added
                try:
                    with processor.db_connection.get_engine().connect() as conn:
                        # Check relationships for this specific document
                        result = conn.execute(text("""
                            SELECT COUNT(*) FROM logical_relationships lr
                            JOIN doc_chunks dc1 ON lr.source_chunk_id = dc1.id
                            WHERE dc1.doc_id = :doc_id
                        """), {"doc_id": doc_id})
                        doc_relationships = result.scalar() or 0
                        
                        # Get total relationships
                        result = conn.execute(text("SELECT COUNT(*) FROM logical_relationships"))
                        rel_count_after = result.scalar() or 0
                        
                        new_rels = rel_count_after - rel_count_before
                        logger.info(f"[3_PROCESS] Added {new_rels} new relationships. Total now: {rel_count_after}")
                        logger.info(f"[3_PROCESS] Document {doc_id} has {doc_relationships} relationships")
                except Exception as e:
                    logger.warning(f"[3_PROCESS] Could not get updated relationship count: {e}")
                
                # Clean up
                await processor.close()
                logger.info(f"[3_PROCESS] HopRAG processing completed for {doc_id}")
                
                # Ensure any remaining changes from HopRAG are committed to the database
                session.commit()
                logger.info(f"[3_PROCESS] Successfully committed HopRAG relationship changes")
                
            except Exception as e:
                session.rollback()  # Roll back any failed HopRAG changes
                logger.error(f"[3_PROCESS] HopRAG processing failed for {doc_id}: {str(e)}")
                import traceback
                logger.error(f"[3_PROCESS] Traceback: {traceback.format_exc()}")
                # Don't fail the entire process if HopRAG fails
            
        except Exception as e:
            session.rollback()
            logger.error(f"[3_PROCESS] Error updating embeddings in database: {e}")
        finally:
            session.close()
        
        logger.info(f"[3_PROCESS] File {file_path} processed successfully")
        
        return db_chunks
    
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.error(f"[3_PROCESS] Error processing file {file_path}: {e}\n\nTraceback:\n{traceback_string}")
        return None

async def process_file_many(file_path): 
    semaphore = asyncio.Semaphore(FILE_PROCESSING_CONCURRENCY)
    async with semaphore:
        return await process_file_one(file_path)

@Logger.log(log_file=project_root / "logs/process.log", log_level="INFO")
async def run_script(force_reprocess: bool = False):
    try:
        logger.warning(f"\n\n[3_PROCESS] Running script with force_reprocess={force_reprocess}...")
        file_paths = get_file_paths()
        
        # Show the progress bar for all file processing
        with tqdm(total=len(file_paths), desc="Processing PDF files", unit="file") as pbar:
            # Create a wrapped version of process_file_many that updates the progress bar
            async def process_with_progress(file_path):
                try:
                    # Pass the force_reprocess parameter to process_file_one
                    result = await process_file_one(file_path, force_reprocess=force_reprocess)
                    # Update progress bar after processing each file
                    pbar.update(1)
                    # Add the filename to the progress bar description when it completes
                    pbar.set_postfix_str(f"Last: {Path(file_path).name}")
                    return result
                except Exception as e:
                    # Still update progress even on error
                    pbar.update(1)
                    # Show error in progress bar
                    pbar.set_postfix_str(f"Error with {Path(file_path).name}")
                    logger.error(f"[3_PROCESS] Error in process_with_progress: {e}")
                    return None
                    
            # Process files with progress tracking
            tasks = [process_with_progress(file_path) for file_path in file_paths]
            chunk_results = await asyncio.gather(*tasks)
        
        # Filter out None results and empty lists
        valid_chunks = [chunks for chunks in chunk_results if chunks is not None and len(chunks) > 0]
        
        if not valid_chunks:
            logger.warning(f"[3_PROCESS] No valid chunks to upload to database")
        else:
            total_chunks = sum(len(chunks) for chunks in valid_chunks)
            logger.warning(f"[3_PROCESS] All files processed successfully. {total_chunks} chunks processed")
            logger.warning(f"[3_PROCESS] Script completed successfully. Exiting.")
    except Exception as e:
        traceback_string = traceback.format_exc()
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 3_process.py: {e}\n\nTraceback: {traceback.format_exc()}\n\n\n\n")
        raise e

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process PDF files and add them to the database")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of all documents, even if already processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the script with force_reprocess parameter
    asyncio.run(run_script(force_reprocess=args.force))