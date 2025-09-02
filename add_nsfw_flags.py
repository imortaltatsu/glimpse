#!/usr/bin/env python3
"""
Batch script to add NSFW flags to existing indexed data
"""

import chromadb
from chromadb.config import Settings
import torch
import numpy as np
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100  # Process in batches to avoid memory issues
NSFW_THRESHOLD = 0.6  # Threshold for NSFW classification

# NSFW and Safe labels for zero-shot classification
NSFW_LABELS = [
    "nude person", "sexual content", "explicit content", "adult content",
    "inappropriate content", "nsfw content", "pornographic content",
    "violence", "graphic content", "disturbing content"
]

SAFE_LABELS = [
    "safe content", "appropriate content", "family friendly content",
    "clean content", "professional content", "educational content",
    "artistic content", "nature content", "technology content"
]

def load_imagebind_model():
    """Load ImageBind model for NSFW detection"""
    print(f"🔄 Loading ImageBind model on {DEVICE}...")
    
    # Show GPU information
    if torch.cuda.is_available():
        print(f"🔍 CUDA available: {torch.cuda.is_available()}")
        print(f"🔍 Device count: {torch.cuda.device_count()}")
        print(f"🔍 Current device: {torch.cuda.current_device()}")
        if "cuda" in DEVICE:
            device_id = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
            print(f"🔍 Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            print(f"🔍 GPU {device_id} memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(DEVICE)
    print("✅ ImageBind model loaded successfully")
    return model

def detect_nsfw_zero_shot(model, content_embedding):
    """
    Detect NSFW content using zero-shot classification
    """
    try:
        # Debug: log the type and shape of the embedding (only for first few entries)
        if hasattr(detect_nsfw_zero_shot, '_debug_count'):
            detect_nsfw_zero_shot._debug_count += 1
        else:
            detect_nsfw_zero_shot._debug_count = 1
            
        if detect_nsfw_zero_shot._debug_count <= 1:
            logger.info(f"Content embedding type: {type(content_embedding)}, length: {len(content_embedding) if isinstance(content_embedding, (list, tuple)) else 'N/A'}")
        # Create text embeddings for NSFW and safe labels
        nsfw_embeddings = []
        safe_embeddings = []
        
        for i, text in enumerate(NSFW_LABELS):
            text_inputs = data.load_and_transform_text([text], DEVICE)
            with torch.no_grad():
                text_embedding = model.modality_preprocessors[ModalityType.TEXT](text_inputs)
                
                # Debug logging for first text embedding only
                if i < 1:
                    logger.info(f"Text embedding type: {type(text_embedding)}")
                    if isinstance(text_embedding, dict):
                        logger.info(f"Text embedding keys: {list(text_embedding.keys())}")
                        if 'trunk' in text_embedding and 'tokens' in text_embedding['trunk']:
                            logger.info(f"  trunk.tokens shape: {text_embedding['trunk']['tokens'].shape}")
                
                # Handle different return types from ImageBind
                if hasattr(text_embedding, 'cpu'):
                    nsfw_embeddings.append(text_embedding.cpu().numpy())
                elif isinstance(text_embedding, dict):
                    # ImageBind returns nested dict structure: {'trunk': {...}, 'head': {...}}
                    # The actual embedding is in trunk.tokens with shape [1, 77, 1024]
                    emb = None
                    
                    if 'trunk' in text_embedding and 'tokens' in text_embedding['trunk']:
                        emb = text_embedding['trunk']['tokens']
                    else:
                        # Fallback: try to find the largest tensor
                        for key, value in text_embedding.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if hasattr(subvalue, 'shape') and hasattr(subvalue, 'numel') and subvalue.numel() > 100:
                                        emb = subvalue
                                        break
                            elif hasattr(value, 'shape') and hasattr(value, 'numel') and value.numel() > 100:
                                emb = value
                                break
                    
                    if emb is not None:
                        if hasattr(emb, 'cpu'):
                            # For tokens with shape [1, 77, 1024], we need to reshape to [1024] for comparison
                            if len(emb.shape) == 3 and emb.shape[0] == 1:
                                emb = emb.squeeze(0).mean(dim=0)  # Average over sequence length
                            nsfw_embeddings.append(emb.cpu().numpy())
                        else:
                            nsfw_embeddings.append(np.array(emb))
                    else:
                        logger.error(f"Could not extract embedding from nested dict structure")
                        continue
                else:
                    nsfw_embeddings.append(np.array(text_embedding))
        
        for text in SAFE_LABELS:
            text_inputs = data.load_and_transform_text([text], DEVICE)
            with torch.no_grad():
                text_embedding = model.modality_preprocessors[ModalityType.TEXT](text_inputs)
                
                # Handle different return types from ImageBind
                if hasattr(text_embedding, 'cpu'):
                    safe_embeddings.append(text_embedding.cpu().numpy())
                elif isinstance(text_embedding, dict):
                    # ImageBind returns nested dict structure: {'trunk': {...}, 'head': {...}}
                    # The actual embedding is in trunk.tokens with shape [1, 77, 1024]
                    emb = None
                    
                    if 'trunk' in text_embedding and 'tokens' in text_embedding['trunk']:
                        emb = text_embedding['trunk']['tokens']
                    else:
                        # Fallback: try to find the largest tensor
                        for key, value in text_embedding.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    if hasattr(subvalue, 'shape') and hasattr(subvalue, 'numel') and subvalue.numel() > 100:
                                        emb = subvalue
                                        break
                            elif hasattr(value, 'shape') and hasattr(value, 'numel') and value.numel() > 100:
                                emb = value
                                break
                    
                    if emb is not None:
                        if hasattr(emb, 'cpu'):
                            # For tokens with shape [1, 77, 1024], we need to reshape to [1024] for comparison
                            if len(emb.shape) == 3 and emb.shape[0] == 1:
                                emb = emb.squeeze(0).mean(dim=0)  # Average over sequence length
                            safe_embeddings.append(emb.cpu().numpy())
                        else:
                            safe_embeddings.append(np.array(emb))
                    else:
                        logger.error(f"Could not extract embedding from nested dict structure")
                        continue
                else:
                    safe_embeddings.append(np.array(text_embedding))
        
        # Calculate similarities - handle different embedding formats
        if hasattr(content_embedding, 'cpu'):
            # PyTorch tensor
            content_embedding_np = content_embedding.cpu().numpy()
        elif isinstance(content_embedding, (list, tuple)):
            # List or tuple from ChromaDB
            content_embedding_np = np.array(content_embedding)
        elif isinstance(content_embedding, dict):
            # Dictionary - extract the embedding values
            if 'embedding' in content_embedding:
                content_embedding_np = np.array(content_embedding['embedding'])
            else:
                # Try to convert dict values to array
                content_embedding_np = np.array(list(content_embedding.values()))
        else:
            # Assume it's already a numpy array
            content_embedding_np = np.array(content_embedding)
        
        # Ensure it's a 1D array
        content_embedding_np = content_embedding_np.flatten()
        
        # Calculate cosine similarity with NSFW labels
        nsfw_similarities = []
        for nsfw_emb in nsfw_embeddings:
            similarity = np.dot(content_embedding_np.flatten(), nsfw_emb.flatten()) / (
                np.linalg.norm(content_embedding_np) * np.linalg.norm(nsfw_emb)
            )
            nsfw_similarities.append(similarity)
        
        # Calculate cosine similarity with safe labels
        safe_similarities = []
        for safe_emb in safe_embeddings:
            similarity = np.dot(content_embedding_np.flatten(), safe_emb.flatten()) / (
                np.linalg.norm(content_embedding_np) * np.linalg.norm(safe_emb)
            )
            safe_similarities.append(similarity)
        
        # Calculate NSFW score
        max_nsfw_similarity = max(nsfw_similarities) if nsfw_similarities else 0
        max_safe_similarity = max(safe_similarities) if safe_similarities else 0
        
        # NSFW score is the difference between max NSFW similarity and max safe similarity
        nsfw_score = max_nsfw_similarity - max_safe_similarity
        
        # Normalize to 0-1 range
        nsfw_score = (nsfw_score + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Determine if content is NSFW
        is_nsfw = nsfw_score > NSFW_THRESHOLD
        
        return {
            "is_nsfw": is_nsfw,
            "nsfw_score": float(nsfw_score),
            "max_nsfw_similarity": float(max_nsfw_similarity),
            "max_safe_similarity": float(max_safe_similarity),
            "confidence": float(abs(max_nsfw_similarity - max_safe_similarity))
        }
        
    except Exception as e:
        logger.error(f"NSFW detection failed: {e}")
        return {
            "is_nsfw": False,
            "nsfw_score": 0.0,
            "max_nsfw_similarity": 0.0,
            "max_safe_similarity": 0.0,
            "confidence": 0.0,
            "error": str(e)
        }

def process_collection(collection, model, collection_name):
    """Process a single collection to add NSFW flags"""
    print(f"\n🔄 Processing collection: {collection_name}")
    
    # Get total count
    total_count = collection.count()
    print(f"📊 Total entries: {total_count}")
    
    if total_count == 0:
        print("⚠️ Collection is empty, skipping...")
        return
    
    # Process in batches
    processed = 0
    nsfw_count = 0
    
    for offset in range(0, total_count, BATCH_SIZE):
        try:
            # Get batch of entries
            results = collection.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["embeddings", "metadatas"]
            )
            
            if not results['ids']:
                break
            
            # Prepare updates
            updated_metadatas = []
            updated_ids = []
            
            for i, (entry_id, metadata, embedding) in enumerate(zip(
                results['ids'], 
                results['metadatas'], 
                results['embeddings']
            )):
                # Skip if already has NSFW flag
                if metadata and 'is_nsfw' in metadata:
                    continue
                
                # Detect NSFW
                nsfw_result = detect_nsfw_zero_shot(model, embedding)
                
                # Update metadata
                updated_metadata = metadata.copy() if metadata else {}
                updated_metadata['is_nsfw'] = nsfw_result['is_nsfw']
                updated_metadata['nsfw_score'] = nsfw_result['nsfw_score']
                updated_metadata['nsfw_confidence'] = nsfw_result['confidence']
                
                updated_metadatas.append(updated_metadata)
                updated_ids.append(entry_id)
                
                if nsfw_result['is_nsfw']:
                    nsfw_count += 1
                
                processed += 1
            
            # Update the collection with new metadata
            if updated_ids:
                collection.update(
                    ids=updated_ids,
                    metadatas=updated_metadatas
                )
                print(f"✅ Updated batch {offset//BATCH_SIZE + 1}: {len(updated_ids)} entries")
            
            # Progress update
            if processed % 1000 == 0:
                print(f"📈 Progress: {processed}/{total_count} ({processed/total_count*100:.1f}%)")
                print(f"🚨 NSFW entries found so far: {nsfw_count}")
            
        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            continue
    
    print(f"✅ Collection {collection_name} complete!")
    print(f"📊 Processed: {processed} entries")
    print(f"🚨 NSFW entries: {nsfw_count} ({nsfw_count/processed*100:.1f}%)")

def main():
    """Main function to process all collections"""
    print("🚀 Starting NSFW flag addition to existing indexed data")
    print(f"🔧 Device: {DEVICE}")
    print(f"📏 Batch size: {BATCH_SIZE}")
    print(f"🎯 NSFW threshold: {NSFW_THRESHOLD}")
    
    # Set CUDA device if using GPU
    if "cuda" in DEVICE:
        device_id = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
        torch.cuda.set_device(device_id)
        print(f"🎯 Set CUDA device to GPU {device_id}")
    
    # Load ImageBind model
    model = load_imagebind_model()
    
    # Initialize ChromaDB client
    print("\n🔄 Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path="index_data/chroma_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    
    # Get all collections
    collections = chroma_client.list_collections()
    print(f"📚 Found {len(collections)} collections")
    
    # Process each collection
    for collection_info in collections:
        collection_name = collection_info.name
        collection = chroma_client.get_collection(collection_name)
        
        # Skip ARNS collection if it's empty
        if collection_name == "arweave_arns" and collection.count() == 0:
            print(f"⏭️ Skipping empty collection: {collection_name}")
            continue
        
        process_collection(collection, model, collection_name)
    
    print("\n🎉 NSFW flag addition complete!")
    print("💡 All entries now have 'is_nsfw' boolean flags in their metadata")

if __name__ == "__main__":
    main()
