import sys
import os 
from pathlib import Path
import traceback
import logging
import asyncio
from dotenv import load_dotenv
from datetime import datetime, date
import uuid
from tqdm import tqdm
from sqlalchemy import select

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from extract_document import extract_text_from_pdf
from chunking import DocChunker
from helpers.internal import Logger
from constants.settings import FILE_PROCESSING_CONCURRENCY
from databases.auth import PostgresConnection
from databases.models import NDCDocumentORM, DocChunkORM, LogicalRelationshipORM
from databases.operations import check_document_processed, update_processed, upload

logger = logging.getLogger(__name__)
db = PostgresConnection()
load_dotenv()


def get_file_paths():
    """
    Get the file paths of PDF files in the data/pdfs folder.
    """
    pdf_directory = project_root / "data" / "pdfs"
    if not pdf_directory.exists():
        logger.warning(f"[3_CHUNK] PDF directory not found: {pdf_directory}")
        return []
    
    pdf_files = list(pdf_directory.glob("*.pdf"))
    logger.info(f"[3_CHUNK] Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    # Convert Path objects to strings
    return [str(file_path) for file_path in pdf_files]


async def chunk_file_one(file_path: str, force_reprocess: bool = False):
    """
    Process a file and create chunks (without embeddings).
    
    Args:
        file_path: Path to the PDF file
        force_reprocess: If True, reprocess the file even if it has been processed before
    """
    try:
        # Initialize database connection
        with db.get_session() as session:
            # Extract file name to be used as document ID
            file_name = Path(file_path).stem
            logger.info(f"[3_CHUNK] Processing file {file_path}")


            # Check if document has already been processed
            doc_id = file_name
            is_processed, document = check_document_processed(session, doc_id)
            
            if is_processed and not force_reprocess:
                logger.info(f"[3_CHUNK] Document {doc_id} has already been processed. Skipping.")
                return None
                
            # If force_reprocess is True and document exists, delete existing chunks and relationships
            if force_reprocess and document:
                logger.info(f"[3_CHUNK] Force reprocessing document {doc_id}")

                try:
                    # Convert string doc_id to UUID using deterministic UUID5
                    doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)
                    
                    
                    # First, delete any logical relationships that reference this document's chunks
                    # Get all chunk IDs for this document using explicit select
                    chunk_ids_query = select(DocChunkORM.id).filter(DocChunkORM.doc_id == doc_uuid)
                    
                    # Delete relationships where either source or target chunk belongs to this document
                    deleted_relationships = session.query(LogicalRelationshipORM).filter(
                        (LogicalRelationshipORM.source_chunk_id.in_(chunk_ids_query)) |
                        (LogicalRelationshipORM.target_chunk_id.in_(chunk_ids_query))
                    ).delete(synchronize_session=False)
                    
                    if deleted_relationships > 0:
                        logger.info(f"[3_CHUNK] Deleted {deleted_relationships} relationships for document {doc_id}")
                    
                    # Now delete the chunks
                    deleted_chunks = session.query(DocChunkORM).filter(DocChunkORM.doc_id == doc_uuid).delete()
                    logger.info(f"[3_CHUNK] Removed {deleted_chunks} existing chunks for document {doc_id}")
                    
                    session.commit()
                except Exception as e:
                    logger.error(f"[3_CHUNK] Error removing existing chunks and relationships: {e}")
                    logger.error(f"[3_CHUNK] Traceback: {traceback.format_exc()}")
                    session.rollback()
                finally:
                    session.close()
            
            # If document doesn't exist, create it first before proceeding with chunk processing
            if not document:
                logger.info(f"[3_CHUNK] Document {doc_id} not found in database. Creating record...")

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
                            logger.warning(f"[3_CHUNK] Could not parse date from filename: {parts[2]}")
                    
                    # Create document record
                    new_document = NDCDocumentORM(
                        doc_id=uuid.uuid5(uuid.NAMESPACE_DNS, doc_id),
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
                    logger.info(f"[3_CHUNK] Created document record for {doc_id}")
                    document = new_document
                except Exception as e:
                    logger.error(f"[3_CHUNK] Error creating document record: {e}")
                    session.rollback()
                finally:
                    session.close()
            
            # 1. Extract text from PDF with multiple strategies, stopping once one succeeds
            extracted_elements = None
            extraction_strategies = ['fast', 'auto', 'ocr_only']
            
            for strategy in extraction_strategies:
                try:
                    logger.info(f"[3_CHUNK] Attempting text extraction with strategy: {strategy}")
                    raw_extracted_elements = extract_text_from_pdf(file_path, strategy=strategy)

                    # Check if we got meaningful content from the extraction function
                    if raw_extracted_elements and len(raw_extracted_elements) > 0:
                        logger.info(f"[3_CHUNK] Strategy {strategy} extracted {len(raw_extracted_elements)} elements")
                        
                        # Handle different element structures - some may be dicts, others may be objects
                        text_content = []
                        for elem in raw_extracted_elements:
                            try:
                                # Try dict-like access first
                                if hasattr(elem, 'get'):
                                    text = elem.get('text', '').strip()
                                    metadata = elem.get('metadata', {})
                                # Try object attribute access
                                elif hasattr(elem, 'text'):
                                    text = getattr(elem, 'text', '').strip()
                                    metadata = getattr(elem, 'metadata', {})
                                # Try direct string conversion
                                elif isinstance(elem, str):
                                    text = elem.strip()
                                    metadata = {}
                                # Try converting object to dict
                                elif hasattr(elem, '__dict__'):
                                    elem_dict = elem.__dict__
                                    text = elem_dict.get('text', '').strip()
                                    metadata = elem_dict.get('metadata', {})
                                else:
                                    text = str(elem).strip()
                                    metadata = {}
                                
                                if text:
                                    # Create standardized element structure
                                    standardized_elem = {
                                        'text': text,
                                        'metadata': metadata if isinstance(metadata, dict) else {}
                                    }
                                    text_content.append(standardized_elem)
                            except Exception as elem_error:
                                logger.warning(f"[3_CHUNK] Error processing element with strategy {strategy}: {elem_error}")
                                continue
                        
                        if text_content:
                            logger.info(f"[3_CHUNK] Successfully extracted {len(text_content)} text elements using {strategy} strategy")
                            extracted_elements = text_content
                            break  # Success! Stop trying other strategies
                        else:
                            logger.warning(f"[3_CHUNK] Strategy {strategy} returned elements but no actual text content")
                    else:
                        logger.warning(f"[3_CHUNK] Strategy {strategy} returned no elements")
                        
                except Exception as e:
                    logger.warning(f"[3_CHUNK] Strategy {strategy} failed: {e}")
                    extracted_elements = None
                    continue
            
            # If all extraction strategies failed, create a minimal fallback
            if not extracted_elements:
                logger.error(f"[3_CHUNK] All text extraction strategies failed for {file_path}")
                
                # Create a minimal fallback chunk with filename information
                fallback_text = f"Document: {Path(file_path).name}\nNote: Text extraction failed - document may be image-based or corrupted."
                
                extracted_elements = [{
                    'text': fallback_text,
                    'metadata': {
                        'page_number': 1,
                        'paragraph_number': 1,
                        'extraction_method': 'fallback',
                        'extraction_status': 'failed'
                    }
                }]
                
                logger.warning(f"[3_CHUNK] Using fallback content for {file_path}")
            
            logger.info(f"[3_CHUNK] Final extracted elements count: {len(extracted_elements)}")
            
            # 2. Create chunks from the extracted text
            doc_chunker = DocChunker()
            chunks = doc_chunker.chunk_document_by_sentences(extracted_elements)
            logger.info(f"[3_CHUNK] Created {len(chunks)} chunks from {file_path}")
            
            # 3. Clean chunks
            try:
                cleaned_chunks = doc_chunker.cleaning_function(chunks)
                logger.info(f"[3_CHUNK] Finished cleaning chunks from {file_path}, got {len(cleaned_chunks)} cleaned chunks")
            except Exception as e:
                logger.error(f"[3_CHUNK] Error during chunk cleaning: {e}")
                # Fall back to using the original chunks if cleaning fails
                logger.info(f"[3_CHUNK] Falling back to original chunks")
                cleaned_chunks = chunks
                
            # Validate cleaned chunks before proceeding
            if not cleaned_chunks or len(cleaned_chunks) == 0:
                logger.error(f"[3_CHUNK] No valid chunks after cleaning for {file_path}")
                return None
            
            # 4. Create DocChunk objects for database storage (without embeddings)
            db_chunks = []
            doc_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, file_name)
            for i, chunk in enumerate(cleaned_chunks):
                # Skip empty chunks
                if not chunk.get('text', '').strip():
                    logger.warning(f"[3_CHUNK] Skipping empty chunk {i} from {file_path}")
                    continue
                    
                # Create a DocChunkORM object without embeddings
                chunk_model = DocChunkORM(
                    id=uuid.uuid4(),
                    doc_id=doc_uuid,
                    content=chunk.get('text', ''),
                    chunk_index=i,
                    page=chunk.get('metadata', {}).get('page_number', 0),
                    paragraph=chunk.get('metadata', {}).get('paragraph_numbers', 0)[0],
                    language=chunk.get('metadata', {}).get('language'),
                    chunk_data=chunk.get('metadata', {}),
                    # Note: transformer_embedding and word2vec_embedding are left as None
                    # They will be filled in by 3.5_embed.py
                )

                db_chunks.append(chunk_model)
            
            # Verify we have chunks to upload
            if not db_chunks:
                logger.error(f"[3_CHUNK] No valid chunks to upload for {file_path}")
                return None
                
            # 5. Upload chunks to the database (without embeddings)
            logger.info(f"[3_CHUNK] Attempting to upload {len(db_chunks)} chunks to database for {file_path}")
            upload_success = upload(session, db_chunks, table='doc_chunks')
            if not upload_success:
                logger.error(f"[3_CHUNK] Failed to upload chunks to database for {file_path}")
                return None
                
            logger.info(f"[3_CHUNK] Uploaded {len(db_chunks)} chunks to doc_chunks table")

            # 6. Update documents table that the document has been processed (chunked)
            update_processed(session, NDCDocumentORM, doc_id, chunks=cleaned_chunks, table='documents')
            logger.info(f"[3_CHUNK] Updated processed status in documents table for {doc_id}")
            
            logger.info(f"[3_CHUNK] File {file_path} chunked successfully")
            return db_chunks
    
    except Exception as e:
        logger.error(f"[3_CHUNK] Error processing file {file_path}: {e}\n\n")
        logger.error(f"[3_CHUNK] Traceback: {traceback.format_exc()}")
        return None


async def chunk_file_many(file_path): 
    """Process a single file with concurrency control."""
    semaphore = asyncio.Semaphore(FILE_PROCESSING_CONCURRENCY)
    async with semaphore:
        return await chunk_file_one(file_path)


@Logger.log(log_file = project_root / "logs/chunk.log", log_level="INFO")
async def run_script(force_reprocess: bool = False):
    """
    Main function to process all PDF files and create chunks.
    
    Args:
        force_reprocess: If True, reprocess all documents even if already processed
    """
    try:
        logger.warning(f"\n\n[3_CHUNK] Running chunking script with force_reprocess={force_reprocess}...")
        file_paths = get_file_paths()
        
        if not file_paths:
            logger.warning(f"[3_CHUNK] No PDF files found to process")
            return
        
        # Show the progress bar for all file processing
        with tqdm(total=len(file_paths), desc="Chunking PDF files", unit="file") as pbar:
            # Create a wrapped version that updates the progress bar
            async def process_with_progress(file_path):
                try:
                    # Pass the force_reprocess parameter to chunk_file_one
                    result = await chunk_file_one(file_path, force_reprocess=force_reprocess)
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
                    logger.error(f"[3_CHUNK] Error in process_with_progress: {e}")
                    return None
                    
            # Process files with progress tracking
            tasks = [process_with_progress(file_path) for file_path in file_paths]
            chunk_results = await asyncio.gather(*tasks)
        
        # Filter out None results and empty lists
        valid_chunks = [chunks for chunks in chunk_results if chunks is not None and len(chunks) > 0]
        
        if not valid_chunks:
            logger.warning(f"[3_CHUNK] No valid chunks created")
        else:
            total_chunks = sum(len(chunks) for chunks in valid_chunks)
            logger.warning(f"[3_CHUNK] All files processed successfully. {total_chunks} chunks created")
            logger.warning(f"[3_CHUNK] Chunking script completed successfully. Ready for embedding.")
            
    except Exception as e:
        logger.critical(f"\n\n\n\n[PIPELINE BROKE!] - Error in 3_chunk.py: {e}\n\n")
        logger.critical(f"[3_CHUNK] Traceback: {traceback.format_exc()}")
        raise e


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Extract text and create chunks from PDF files")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing of all documents, even if already processed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the script with force_reprocess parameter
    asyncio.run(run_script(force_reprocess=args.force))