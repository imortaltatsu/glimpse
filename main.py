# Complete Multimodal Arweave Indexer using ImageBind + cuML + ChromaDB

import os
import io
import json
import time
import threading
import requests
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torchaudio
import av
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import load_and_transform_audio_data
from cuml.neighbors import NearestNeighbors
import mimetypes
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
import librosa
import subprocess
import uuid
import re
import chromadb
from chromadb.config import Settings

print("✅ All imports completed successfully")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Add file handler for logging to 'indexer.log'
file_handler = logging.FileHandler('indexer.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
logger.addHandler(file_handler)

# ==== Config ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "index_data"
os.makedirs(DATA_DIR, exist_ok=True)
MODALITIES = ["web", "image", "audio", "video", "all"]

INDEX_FILES = {m: os.path.join(DATA_DIR, f"{m}_index.npy") for m in MODALITIES}
META_FILES = {m: os.path.join(DATA_DIR, f"{m}_meta.json") for m in MODALITIES}
CURSOR_FILE = os.path.join(DATA_DIR, "last_cursor.txt")
BATCH_SIZE = 100
TOP_K = 10
POLL_INTERVAL = 60

# ==== ChromaDB Configuration ====
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create collections for each modality
collections = {}
for modality in MODALITIES:
    try:
        collections[modality] = chroma_client.get_or_create_collection(
            name=f"arweave_{modality}",
            metadata={"description": f"Arweave {modality} embeddings"}
        )
        logger.info(f"✅ ChromaDB collection 'arweave_{modality}' ready")
    except Exception as e:
        logger.error(f"Failed to create ChromaDB collection for {modality}: {e}")
        collections[modality] = None

# Create ARNS-specific collection
try:
    collections["arns"] = chroma_client.get_or_create_collection(
        name="arweave_arns",
        metadata={"description": "Arweave ARNS embeddings"}
    )
    logger.info("✅ ChromaDB collection 'arweave_arns' ready")
except Exception as e:
    logger.error(f"Failed to create ARNS ChromaDB collection: {e}")
    collections["arns"] = None

# ==== Load ImageBind ====
print("🔄 Loading ImageBind model...")
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(DEVICE)
print(f"✅ ImageBind model loaded successfully on {DEVICE}")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
print("✅ Text splitter initialized")

# ==== Index Handling ====
def load_index(path):
    try:
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape[1] != 1024:
                logger.warning(f"Index file {path} has wrong shape {arr.shape}, recreating.")
                return np.empty((0, 1024), dtype=np.float32)
            return arr
        else:
            return np.empty((0, 1024), dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to load index file {path}: {e}. Recreating empty index.")
        return np.empty((0, 1024), dtype=np.float32)

def save_index(index, path):
    try:
        np.save(path, index)
        logger.info(f"Index file saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save index file {path}: {e}")

def save_meta(meta, path):
    try:
        with open(path, "w") as f:
            json.dump(meta, f)
        logger.info(f"Meta file saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save meta file {path}: {e}")

def load_meta(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

def save_cursor(cursor, modality):
    with open(os.path.join(DATA_DIR, f"{modality}_cursor.txt"), "w") as f:
        f.write(cursor or "")

def load_cursor(modality):
    path = os.path.join(DATA_DIR, f"{modality}_cursor.txt")
    return open(path).read().strip() if os.path.exists(path) else None

# ==== ChromaDB Storage Functions ====
def store_in_chromadb(embedding, metadata, modality):
    """Store embedding and metadata in ChromaDB collection."""
    try:
        if modality not in collections or collections[modality] is None:
            logger.error(f"ChromaDB collection for {modality} not available")
            return False
        
        # Generate unique ID for the document
        doc_id = f"{modality}_{metadata.get('txid', str(uuid.uuid4()))}"
        
        # Add embedding to collection
        collections[modality].add(
            embeddings=[embedding.tolist()],
            documents=[metadata.get('chunk', '')],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        logger.info(f"✅ Stored embedding in ChromaDB for {modality}: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store in ChromaDB for {modality}: {e}")
        return False

def search_in_chromadb(query_embedding, modality, top_k=10):
    """Search for similar embeddings in ChromaDB collection."""
    try:
        if modality not in collections or collections[modality] is None:
            logger.error(f"ChromaDB collection for {modality} not available")
            return {"results": []}
        
        # Search in collection
        results = collections[modality].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    "score": results['distances'][0][i],  # Convert distance to similarity
                    "id": results['ids'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "document": results['documents'][0][i]
                }
                formatted_results.append(result)
        
        logger.info(f"✅ ChromaDB search for {modality} returned {len(formatted_results)} results")
        return {"results": formatted_results}
        
    except Exception as e:
        logger.error(f"Failed to search in ChromaDB for {modality}: {e}")
        return {"results": []}

def get_collection_count(modality):
    """Get the number of documents in a ChromaDB collection."""
    try:
        if modality not in collections or collections[modality] is None:
            return 0
        return collections[modality].count()
    except Exception as e:
        logger.error(f"Failed to get count for {modality}: {e}")
        return 0
        
# Example: a larger list (expand as needed)
ARWEAVE_DOMAINS = [
    "arweave.net", "arnode.asia", "ar.io", "arweave.dev", "arweave.live", "arweave-search.goldsky.com"
    # ...add more from the community list
]

from urllib.parse import urlparse, urlunparse

def arweave_domain_fallback_urls(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    # If it's a subdomain (e.g., foo.arweave.net)
    for domain in ARWEAVE_DOMAINS:
        if any(hostname.endswith(f'.{d}') for d in ARWEAVE_DOMAINS):
            # Replace the root domain, keep the subdomain prefix
            parts = hostname.split(".")
            if len(parts) > 2:
                sub = ".".join(parts[:-2])
                new_host = f"{sub}.{domain}"
            else:
                new_host = domain
            new_url = urlunparse(parsed._replace(netloc=new_host))
            yield new_url
        else:
            # Root-level, just swap the domain
            new_url = urlunparse(parsed._replace(netloc=domain))
            yield new_url


RETRY_COUNT = 3
RETRY_SLEEP = 2  # seconds

def get_magic_bytes(url, num_bytes=16):
    if "cu.ardrive.io" in url:
        for attempt in range(RETRY_COUNT):
            try:
                resp = requests.get(url, stream=True, timeout=15)
                chunk = resp.raw.read(num_bytes)
                resp.close()
                return chunk
            except Exception as e:
                logger.warning(f"[get_magic_bytes] Attempt {attempt+1} failed for {url}: {e}")
                time.sleep(RETRY_SLEEP)
        return b""
    for test_url in arweave_domain_fallback_urls(url):
        for attempt in range(RETRY_COUNT):
            try:
                resp = requests.get(test_url, stream=True, timeout=15)
                chunk = resp.raw.read(num_bytes)
                resp.close()
                return chunk
            except Exception as e:
                logger.warning(f"[get_magic_bytes] Attempt {attempt+1} failed for {test_url}: {e}")
                time.sleep(RETRY_SLEEP)
    return b""

# --- Improved File Type Detection ---
def detect_file_type(url):
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    if ext == '.pdf':
        return "pdf"
    # Content-Type header check
    for domain in ARWEAVE_DOMAINS:
        try:
            test_url = url
            for d in ARWEAVE_DOMAINS:
                if d in url:
                    test_url = url.replace(d, domain)
                    break
            resp = requests.head(test_url, timeout=10, allow_redirects=True)
            content_type = resp.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type:
                return "pdf"
            if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                return "web"
            if 'image/' in content_type:
                return "image"
            if 'audio/' in content_type:
                return "audio"
            if 'video/' in content_type:
                return "video"
        except Exception:
            continue
    # Magic bytes check for PDF
    magic = get_magic_bytes(url, 64)
    if magic.startswith(b'%PDF'):
        return "pdf"
    # 1. Check extension first for web types
    if ext in ['.html', '.htm', '.xml', '.json', '.txt']:
        return "web"
    if ext in ['.mp3', '.wav', '.wave', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.webm']:
        return "audio"
    if ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.3gp', '.m4v']:
        return "video"
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg']:
        return "image"
    # 2. Check Content-Type header for HTML (try all fallback domains)
    for domain in ARWEAVE_DOMAINS:
        try:
            test_url = url
            for d in ARWEAVE_DOMAINS:
                if d in url:
                    test_url = url.replace(d, domain)
                    break
            resp = requests.head(test_url, timeout=10, allow_redirects=True)
            content_type = resp.headers.get('content-type', '').lower()
            if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                return "web"
            if 'image/' in content_type:
                return "image"
            if 'audio/' in content_type:
                return "audio"
            if 'video/' in content_type:
                return "video"
        except Exception:
            continue
    # 3. Fallback to magic bytes if extension and content-type are missing or ambiguous
    magic = get_magic_bytes(url, 64)
    # Images
    if (magic.startswith(b'\x89PNG') or magic.startswith(b'\xff\xd8\xff') or 
        magic.startswith(b'GIF87a') or magic.startswith(b'GIF89a') or
        magic.startswith(b'BM') or magic.startswith(b'II*\x00') or magic.startswith(b'MM\x00*') or
        (magic[:4] == b'RIFF' and magic[8:12] == b'WEBP') or 
        magic.startswith(b'\x00\x00\x01\x00') or magic.startswith(b'\x00\x00\x02\x00')):
        return "image"
    # Audio
    if (magic[:3] == b'ID3' or magic[:2] == b'\xff\xfb' or magic[:2] == b'\xff\xf3' or
        magic[:2] == b'\xff\xf1' or magic[:2] == b'\xff\xf9' or
        (magic[:4] == b'RIFF' and magic[8:12] == b'WAVE') or
        magic.startswith(b'OggS') or magic.startswith(b'fLaC') or
        magic.startswith(b'FORM') or magic.startswith(b'FLAC') or
        b'ftypM4A' in magic or b'ftypmp4' in magic or
        magic.startswith(b'\xff\xfe') or magic.startswith(b'\xfe\xff') or
        magic.startswith(b'UklGRg==') or magic.startswith(b'SUQzBAAAAAA')):
        return "audio"
    # Video
    if ((magic[:4] == b'\x00\x00\x00\x18' or magic[:4] == b'\x00\x00\x00\x20') and 
         (magic[4:8] == b'ftyp' or b'ftyp' in magic[:16])) or \
       b'ftypqt' in magic or b'ftypisom' in magic or b'ftypmp4' in magic or \
       (magic[:4] == b'RIFF' and magic[8:12] == b'AVI ') or \
       magic.startswith(b'\x1A\x45\xDF\xA3') or magic.startswith(b'FLV') or \
       magic.startswith(b'0&\xB2u\x8Ef\xCF\x11') or \
       magic.startswith(b'\x00\x00\x00\x14') or magic.startswith(b'\x00\x00\x00\x1C'):
        return "video"
    # Web/HTML content (robust check: anywhere in first 64 bytes)
    if (b'<html' in magic.lower() or b'<!doctype' in magic.lower() or b'<HTML' in magic or b'<?xml' in magic or
        magic.startswith(b'{') or magic.startswith(b'[')):
        return "web"
    return "binary"

# --- Helper: Check if URL is a webpage ---
def check_webpage(url):
    try:
        resp = requests.get(url, timeout=5)
        text = resp.text
        if "<!doctype html" in text.lower() or "<html" in text.lower():
            return True
        if any(tag in text.lower() for tag in ["<head", "<body", "<title", "<meta"]):
            return True
        return False
    except Exception:
        return False

# --- Helper: Robots.txt and Crawl Delay ---
def get_robots_txt_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

def parse_robots_txt(robots_url):
    try:
        from urllib.robotparser import RobotFileParser
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp
    except Exception as e:
        logger.warning(f"Failed to parse robots.txt from {robots_url}: {e}")
        return None

def can_fetch_url(url, user_agent="ArweaveIndexer/1.0"):
    try:
        robots_url = get_robots_txt_url(url)
        rp = parse_robots_txt(robots_url)
        if rp is None:
            return None
        return rp.can_fetch(user_agent, url)
    except Exception:
        return None

def get_crawl_delay(url, user_agent="ArweaveIndexer/1.0"):
    try:
        robots_url = get_robots_txt_url(url)
        rp = parse_robots_txt(robots_url)
        if rp is None:
            return 0
        delay = rp.crawl_delay(user_agent)
        return delay if delay is not None else 0
    except Exception:
        return 0

# --- Helper: Extract main content from soup ---
def extract_main_content(soup):
    # Try to find <main> tag first
    main = soup.find('main')
    if main:
        return main
    # Fallback: use <body>
    body = soup.find('body')
    if body:
        return body
    # Fallback: use whole soup
    return soup

# --- Helper: Validate content ---
def is_valid_content(text, min_length=50, max_length=100000):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_length:
        return False
    if len(text) > max_length:
        return False
    if all(c in '\n\r\t ' for c in text):
        return False
    return True

# --- Helper: Clean text ---
def clean_text(text):
    if not text:
        return ""
    return ' '.join(text.split())

# --- Helper: Create semantic chunks from text elements ---
def create_semantic_chunks(text_elements, max_chunk_size=1000, min_chunk_size=100):
    chunks = []
    current_chunk = []
    current_length = 0
    for elem in text_elements:
        text = elem['text']
        if not text:
            continue
        if current_length + len(text) > max_chunk_size and current_chunk:
            chunks.append({'text': ' '.join([e['text'] for e in current_chunk]), 'elements': current_chunk})
            current_chunk = []
            current_length = 0
        current_chunk.append(elem)
        current_length += len(text)
    if current_chunk:
        chunks.append({'text': ' '.join([e['text'] for e in current_chunk]), 'elements': current_chunk})
    # Filter out too-small chunks
    return [c for c in chunks if len(c['text']) >= min_chunk_size]

# --- Helper: Extract enhanced metadata from web loader ---
def extract_enhanced_metadata_from_web_loader(web_loader_metadata, url):
    # This can be expanded for more fields as needed
    return {
        'title': web_loader_metadata.get('title', ''),
        'description': web_loader_metadata.get('description', ''),
        'url': web_loader_metadata.get('url', url),
        'language': web_loader_metadata.get('language', ''),
        'source': web_loader_metadata.get('source', ''),
        'content': web_loader_metadata.get('content', ''),
    }

# --- Helper: Create weighted chunks from web loader ---
def create_weighted_chunks_from_web_loader(web_loader_metadata, enhanced_metadata):
    # For now, just create a single chunk with all content
    content = web_loader_metadata.get('content', '')
    title = web_loader_metadata.get('title', '')
    description = web_loader_metadata.get('description', '')
    chunks = []
    if title:
        chunks.append({'text': title, 'weight': 3.0, 'type': 'title', 'section': 'Title'})
    if description:
        chunks.append({'text': description, 'weight': 2.0, 'type': 'description', 'section': 'Description'})
    if content:
        # Split content into paragraphs for chunking
        for para in content.split('\n'):
            para = para.strip()
            if is_valid_content(para, min_length=50):
                chunks.append({'text': para, 'weight': 1.0, 'type': 'content', 'section': 'Content'})
    return chunks

# --- Helper: Create weighted embeddings for chunks ---
def create_weighted_embeddings(chunks, url, is_arns=False):
    results = []
    for chunk in chunks:
        try:
            emb = embed_text(chunk['text'])
            meta = {
                'txid': extract_txid_or_arns_name(url),
                'url': url,
                'title': chunk.get('section', chunk.get('type', '')),
                'chunk': chunk['text'],
                'description': '',
                'modality': 'web',
                'section': chunk.get('section', ''),
                'weight': chunk.get('weight', 1.0),
                'type': chunk.get('type', 'content'),
            }
            results.append((emb, meta))
        except Exception as e:
            logger.error(f"Failed to embed chunk for {url}: {e}")
    return results

# --- Enhanced Webpage Indexing with Metadata ---
def enhanced_webpage_indexing_with_metadata(url, is_arns=False):
    try:
        if not check_webpage(url):
            logger.info(f"Skipping {url} - not a valid webpage")
            return []
        can_fetch = can_fetch_url(url)
        if can_fetch is False:
            logger.warning(f"Skipping {url} - robots.txt disallows crawling")
            return []
        elif can_fetch is True:
            crawl_delay = get_crawl_delay(url)
            if crawl_delay > 0:
                logger.info(f"Applying crawl delay of {crawl_delay} seconds for {url}")
                time.sleep(crawl_delay)
        web_loader_metadata = None
        web_loader_content = None
        for test_url in arweave_domain_fallback_urls(url):
            try:
                logger.info(f"Attempting WebBaseLoader for {test_url}")
                loader = WebBaseLoader(test_url)
                docs = loader.load()
                if docs:
                    content = "\n".join([doc.page_content for doc in docs])
                    first_doc = docs[0]
                    title = first_doc.metadata.get("title", "") if hasattr(first_doc, 'metadata') else ""
                    description = first_doc.metadata.get("description", "") if hasattr(first_doc, 'metadata') else ""
                    web_loader_metadata = {
                        "title": title,
                        "description": description,
                        "content": content,
                        "source": test_url,
                        "language": first_doc.metadata.get("language", ""),
                        "url": test_url
                    }
                    web_loader_content = content
                    if title.strip() == "ArNS - Arweave Name System":
                        logger.info(f"Skipping {test_url} - standard unassigned ARNS page")
                        return []
                    logger.info(f"✅ WebBaseLoader successful for {test_url}")
                    break
            except Exception as e:
                logger.warning(f"WebBaseLoader failed for {test_url}: {e}")
                continue
        if not web_loader_metadata:
            logger.info(f"WebBaseLoader failed, trying BeautifulSoup fallback for {url}")
            can_fetch_fallback = can_fetch_url(url)
            if can_fetch_fallback is False:
                logger.warning(f"Skipping {url} - robots.txt disallows crawling (fallback)")
                return []
            elif can_fetch_fallback is True:
                crawl_delay = get_crawl_delay(url)
                if crawl_delay > 0:
                    logger.info(f"Applying crawl delay of {crawl_delay} seconds for {url} (fallback)")
                    time.sleep(crawl_delay)
            html_content = None
            for test_url in arweave_domain_fallback_urls(url):
                try:
                    resp = requests.get(test_url, timeout=30, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    resp.raise_for_status()
                    html_content = resp.text
                    break
                except Exception as e:
                    logger.warning(f"Direct fetch failed for {test_url}: {e}")
                    continue
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                main_content = extract_main_content(soup)
                content = main_content.get_text(" ", strip=True) if main_content else ""
                title = ""
                title_selectors = [
                    'meta[property="og:title"]',
                    'meta[name="twitter:title"]', 
                    'meta[name="title"]',
                    'title'
                ]
                for selector in title_selectors:
                    tag = soup.select_one(selector)
                    if tag:
                        if tag.name == 'meta':
                            title = tag.get('content', '').strip()
                        else:
                            title = tag.get_text(strip=True)
                        if title and is_valid_content(title):
                            break
                description = ""
                desc_selectors = [
                    'meta[name="description"]',
                    'meta[property="og:description"]',
                    'meta[name="twitter:description"]'
                ]
                for selector in desc_selectors:
                    tag = soup.select_one(selector)
                    if tag and tag.get('content'):
                        description = tag['content'].strip()
                        if is_valid_content(description):
                            break
                web_loader_metadata = {
                    "title": title,
                    "description": description,
                    "content": content,
                    "source": url,
                    "language": "",
                    "url": url
                }
                web_loader_content = content
                if title.strip() == "ArNS - Arweave Name System":
                    logger.info(f"Skipping {url} - standard unassigned ARNS page")
                    return []
                logger.info(f"✅ BeautifulSoup fallback successful for {url}")
        if web_loader_metadata and web_loader_content:
            enhanced_metadata = extract_enhanced_metadata_from_web_loader(web_loader_metadata, url)
            chunks = create_weighted_chunks_from_web_loader(web_loader_metadata, enhanced_metadata)
            if chunks:
                return create_weighted_embeddings(chunks, url, is_arns)
        logger.warning(f"No valid content extracted from {url}")
        return []
    except Exception as e:
        logger.error(f"Enhanced webpage indexing failed for {url}: {e}")
        return []

# ==== Embedding Functions ====
def embed_text(text):
    inputs = {ModalityType.TEXT: data.load_and_transform_text([text], device=DEVICE)}
    with torch.no_grad():
        emb = model(inputs)[ModalityType.TEXT]
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

def embed_image(url):
    try:
        img_bytes = requests.get(url, timeout=30).content
        temp_dir = os.path.join(DATA_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        filename = url.split("/")[-1]
        if not filename or "." not in filename:
            filename = f"{filename}.jpg"
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(img_bytes)
        logger.info(f"load image from {temp_path}")
        try:
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data([temp_path], device=DEVICE)}
            with torch.no_grad():
                emb = model(inputs)[ModalityType.VISION]
                emb /= emb.norm(dim=-1, keepdim=True)
            return emb[0].cpu().numpy()
        finally:
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"[embed_image] Failed to embed image from {url}: {e}")
        return None

def embed_audio(url):
    try:
        audio_bytes = requests.get(url, timeout=30).content
        temp_dir = os.path.join(DATA_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        unique_id = str(uuid.uuid4())
        temp_in_path = os.path.join(temp_dir, f"input_audio_{unique_id}")
        temp_wav_path = os.path.join(temp_dir, f"temp_{unique_id}.wav")
        # Save the original audio bytes to a temp file
        with open(temp_in_path, 'wb') as f:
            f.write(audio_bytes)
        try:
            # Load and decode audio with librosa (handles most formats)
            y, sr = librosa.load(temp_in_path, sr=None, mono=True)
            # Save as wav using soundfile
            sf.write(temp_wav_path, y, sr, format='WAV')
            logger.info(f"Audio converted and saved as WAV: {temp_wav_path}")
            # Now use your embedding logic on temp_wav_path
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([temp_wav_path], device=DEVICE)}
            with torch.no_grad():
                emb = model(inputs)[ModalityType.AUDIO]
                emb /= emb.norm(dim=-1, keepdim=True)
            return emb[0].cpu().numpy()
        finally:
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
    except Exception as e:
        logger.error(f"[embed_audio] Failed to embed audio from {url}: {e}")
        return None

def ensure_tensor(data):
    # Recursively convert nested lists/arrays to a torch.Tensor
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)) and len(data) > 0:
        return ensure_tensor(data[0])
    return None

def embed_video(url):
    try:
        video_bytes = requests.get(url, timeout=30).content
        temp_dir = os.path.join(DATA_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        filename = url.split("/")[-1]
        if not filename or "." not in filename:
            filename = f"{filename}.mp4"
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
        logger.info(f"load video from {temp_path}")
        try:
            vision_data = None
            if filename.lower().endswith('.mov'):
                mp4_path = temp_path.rsplit('.', 1)[0] + '.mp4'
                try:
                    subprocess.run(['ffmpeg', '-y', '-i', temp_path, mp4_path], check=True)
                    vision_data = data.load_and_transform_video_data([mp4_path], device=DEVICE)
                    os.remove(mp4_path)
                except Exception as conv_e:
                    logger.error(f"[embed_video] MOV to MP4 conversion failed for {url}: {conv_e}")
                    return None
            else:
                vision_data = data.load_and_transform_video_data([temp_path], device=DEVICE)
            # Convert to tensor if needed
            if isinstance(vision_data, np.ndarray):
                vision_data = torch.from_numpy(vision_data)
            if not torch.is_tensor(vision_data):
                try:
                    vision_data = torch.from_numpy(np.array(vision_data))
                except Exception as e:
                    logger.error(f"[embed_video] Could not convert vision_data to tensor for {url}: {e}")
                    return None
            if hasattr(vision_data, 'to'):
                vision_data = vision_data.to(DEVICE)
            inputs = {ModalityType.VISION: vision_data}
            with torch.no_grad():
                emb = model(inputs)[ModalityType.VISION]
                emb /= emb.norm(dim=-1, keepdim=True)
            return emb[0].cpu().numpy()
        finally:
            os.remove(temp_path)
    except Exception as e:
        logger.error(f"[embed_video] Failed to embed video from {url}: {e}")
        return None

# ==== Redundant Fetch Helpers ====
def fetch_with_redundancy(path, method="get", is_arns=False, **kwargs):
    errors = []
    for domain in ARWEAVE_DOMAINS:
        try:
            if is_arns:
                url = f"https://arns.{domain}{path}"
            else:
                url = f"https://{domain}{path}"
            if method == "post":
                resp = requests.post(url, timeout=20, **kwargs)
            else:
                resp = requests.get(url, timeout=20, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            errors.append(f"{domain}: {e}")
    raise Exception(f"All domains failed: {errors}")

def fetch_graphql(query, variables=None):
    for domain in ARWEAVE_DOMAINS:
        url = f"https://{domain}/graphql"
        for attempt in range(RETRY_COUNT):
            try:
                resp = requests.post(url, json={"query": query, "variables": variables or {}}, timeout=20)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"GraphQL fetch attempt {attempt+1} failed on {domain}: {e}")
                time.sleep(RETRY_SLEEP)
    raise Exception("All GraphQL endpoints failed")

def fetch_file(txid):
    for domain in ARWEAVE_DOMAINS:
        url = f"https://{domain}/{txid}"
        for test_url in arweave_domain_fallback_urls(url):
            for attempt in range(RETRY_COUNT):
                try:
                    resp = requests.get(test_url, timeout=30)
                    resp.raise_for_status()
                    return resp.content
                except Exception as e:
                    logger.warning(f"File fetch attempt {attempt+1} failed on {test_url}: {e}")
                    time.sleep(RETRY_SLEEP)
    raise Exception("All file endpoints failed")

def fetch_arns(name):
    for domain in ARWEAVE_DOMAINS:
        url = f"https://arns.{domain}/{name}"
        for test_url in arweave_domain_fallback_urls(url):
            for attempt in range(RETRY_COUNT):
                try:
                    resp = requests.get(test_url, timeout=20)
                    resp.raise_for_status()
                    return resp.content
                except Exception as e:
                    logger.warning(f"ARNS fetch attempt {attempt+1} failed on {test_url}: {e}")
                    time.sleep(RETRY_SLEEP)
    raise Exception("All ARNS endpoints failed")

# ==== Content Parsing (update fetch_webpage_text to use redundancy) ====
def fetch_webpage_text(url):
    if "cu.ardrive.io" in url:
        for attempt in range(RETRY_COUNT):
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                title = docs[0].metadata.get("title", url) if docs and hasattr(docs[0], 'metadata') else url
                description = content[:200] if content else ""
                logger.info(f"Fetched and parsed webpage: {url} (title: {title})")
                return title, content, description
            except Exception as e:
                logger.warning(f"[Fallback] Attempt {attempt+1} failed for cu.ardrive.io: {e}")
                time.sleep(RETRY_SLEEP)
        return "", "", ""
    for test_url in arweave_domain_fallback_urls(url):
        for attempt in range(RETRY_COUNT):
            try:
                loader = WebBaseLoader(test_url)
                docs = loader.load()
                content = "\n".join([doc.page_content for doc in docs])
                title = docs[0].metadata.get("title", test_url) if docs and hasattr(docs[0], 'metadata') else test_url
                description = content[:200] if content else ""
                logger.info(f"Fetched and parsed webpage: {test_url} (title: {title})")
                return title, content, description
            except Exception as e:
                logger.warning(f"[Fallback] Attempt {attempt+1} failed for {test_url}: {e}")
                time.sleep(RETRY_SLEEP)
    return "", "", ""

def extract_txid_or_arns_name(url):
    # Match either a txid (43+ chars) or an ARNS name (subdomain)
    # Example: https://arweave.net/<txid>
    m = re.match(r"https?://(?:[a-zA-Z0-9_-]+\\.)?arweave\\.(?:net|asia|io|dev|live|search\\.goldsky\\.com)/([a-zA-Z0-9_-]{43,})", url)
    if m:
        return m.group(1)
    # Example: https://<name>.arweave.net
    m = re.match(r"https?://([a-zA-Z0-9_-]+)\\.arweave\\.(?:net|asia|io|dev|live|search\\.goldsky\\.com)", url)
    if m:
        return m.group(1)
    return url  # fallback: use the whole url as unique id

def normalize_url(url):
    parsed = urlparse(url)
    # Remove query and fragment, lowercase scheme and netloc, strip trailing slashes
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip('/'),
        params='',
        query='',
        fragment=''
    )
    return urlunparse(normalized)

def parse_webpage(html):
    soup = BeautifulSoup(html, "html.parser")
    # Remove unnecessary tags
    for tag in soup([
        "script", "style", "noscript", "header", "footer", "nav", "aside", "form", "input", "svg", "canvas", "iframe", "button", "figure", "img", "link", "meta", "object", "embed", "applet", "base", "map", "area", "track", "audio", "video"
    ]):
        tag.decompose()
    # Extract SEO meta tags
    def get_meta_content(names):
        for name in names:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
            tag = soup.find("meta", attrs={"property": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""
    title = get_meta_content(["og:title", "twitter:title"]) or (soup.title.string.strip() if soup.title and soup.title.string else "")
    meta_desc = get_meta_content(["description", "og:description", "twitter:description"])
    # Section-aware chunking, clean whitespace
    chunks = []
    current_heading = title
    for elem in soup.find_all(["h1", "h2", "h3", "p"]):
        if elem.name in ["h1", "h2", "h3"]:
            current_heading = elem.get_text(strip=True)
        elif elem.name == "p":
            text = elem.get_text(" ", strip=True)
            text = ' '.join(text.split())  # Remove extra spaces
            if text:
                chunk = {
                    "title": title,
                    "section": current_heading,
                    "text": text,
                    "meta_desc": meta_desc
                }
                chunks.append(chunk)
    return chunks

# ARNS-exclusive index/meta
ARNS_INDEX_FILE = os.path.join(DATA_DIR, "arns_index.npy")
ARNS_META_FILE = os.path.join(DATA_DIR, "arns_meta.json")
def load_arns_index():
    try:
        if os.path.exists(ARNS_INDEX_FILE):
            arr = np.load(ARNS_INDEX_FILE)
            if arr.shape[1] != 1024:
                logger.warning(f"ARNS index file has wrong shape {arr.shape}, recreating.")
                return np.empty((0, 1024), dtype=np.float32)
            return arr
        else:
            return np.empty((0, 1024), dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to load ARNS index: {e}. Recreating empty index.")
        return np.empty((0, 1024), dtype=np.float32)
def load_arns_meta():
    try:
        if os.path.exists(ARNS_META_FILE):
            with open(ARNS_META_FILE) as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to load ARNS meta: {e}. Recreating empty meta.")
        return {}
def save_arns_index(index):
    try:
        np.save(ARNS_INDEX_FILE, index)
        logger.info(f"ARNS index file saved: {ARNS_INDEX_FILE}")
    except Exception as e:
        logger.error(f"Failed to save ARNS index: {e}")
def save_arns_meta(meta):
    try:
        with open(ARNS_META_FILE, "w") as f:
            json.dump(meta, f)
        logger.info(f"ARNS meta file saved: {ARNS_META_FILE}")
    except Exception as e:
        logger.error(f"Failed to save ARNS meta: {e}")

# ChromaDB handles ARNS storage automatically

def index_file(url, is_arns=False, force_web=False):
    return enhanced_index_file(url, is_arns, force_web)

# --- Enhanced Web Indexing and Retrieval (copied from working multimodal ver.py) ---

def enhanced_index_file(url, is_arns=False, force_web=False):
    """Enhanced file indexing with better web content handling."""
    normalized_url = normalize_url(url)
    # Deduplication: skip if normalized_url already indexed
    with indexed_txids_lock:
        if normalized_url in indexed_txids:
            logger.info(f"Skipping already indexed normalized URL: {normalized_url}")
            return
    if force_web:
        filetype = "web"
    else:
        filetype = detect_file_type(url)
    if filetype == "pdf":
        logger.info(f"Skipping PDF file: {url}")
        return
    if filetype == "image":
        emb, modality = embed_image(url), "image"
        if emb is not None:
            meta = {"txid": extract_txid_or_arns_name(url), "url": url, "title": url, "chunk": "", "description": f"{modality} from {url}", "modality": modality}
            store_in_chromadb(emb, meta, modality)
            logger.info(f"Indexed {modality} file: {url}")
        else:
            logger.error(f"Embedding failed for {modality} file: {url}")
        return
    elif filetype == "audio":
        emb, modality = embed_audio(url), "audio"
        if emb is not None:
            meta = {"txid": extract_txid_or_arns_name(url), "url": url, "title": url, "chunk": "", "description": f"{modality} from {url}", "modality": modality}
            store_in_chromadb(emb, meta, modality)
            logger.info(f"Indexed {modality} file: {url}")
        else:
            logger.error(f"Embedding failed for {modality} file: {url}")
        return
    elif filetype == "video":
        emb, modality = embed_video(url), "video"
        if emb is not None:
            meta = {"txid": extract_txid_or_arns_name(url), "url": url, "title": url, "chunk": "", "description": f"{modality} from {url}", "modality": modality}
            store_in_chromadb(emb, meta, modality)
            logger.info(f"Indexed {modality} file: {url}")
        else:
            logger.error(f"Embedding failed for {modality} file: {url}")
        return
    elif filetype == "web":
        try:
            weighted_results = enhanced_webpage_indexing_with_metadata(url, is_arns)
            if not weighted_results:
                logger.warning(f"No valid content extracted from {url}")
                return
            logger.info(f"Indexing {len(weighted_results)} enhanced web chunks from {url}")
            def store_weighted_result(result):
                try:
                    emb, meta = result
                    store_in_chromadb(emb, meta, "web")
                except Exception as e:
                    logger.error(f"Error storing enhanced web result: {e}")
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(store_weighted_result, result) for result in weighted_results]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in enhanced web processing: {e}")
            logger.info(f"Successfully indexed web file with rich metadata: {url}")
        except Exception as e:
            logger.error(f"Web indexing failed for {url}: {e}")
        return
    else:
        logger.info(f"Skipping unsupported or binary file: {url} (detected type: {filetype})")
        return



def enhanced_search_modality(query: str, top_k: int, modality: str):
    logger.info(f"Enhanced search called for modality '{modality}' with query '{query}' and top_k={top_k}")
    try:
        collection = collections.get(modality)
        if not collection:
            logger.warning(f"No ChromaDB collection for modality '{modality}'")
            return {"results": []}
        enhanced_query = query.strip()
        if modality == "web":
            enhanced_query = f"Search query: {enhanced_query}"
        query_embedding = embed_text(enhanced_query)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 5,
            include=["metadatas", "documents", "distances"]
        )
        if not results['ids'] or not results['ids'][0]:
            logger.warning(f"No results found for modality '{modality}'")
            return {"results": []}
        raw_results = []
        for i, (id_val, metadata, document, distance) in enumerate(zip(
            results['ids'][0], 
            results['metadatas'][0], 
            results['documents'][0], 
            results['distances'][0]
        )):
            if metadata and document:
                score = 1.0 - (distance / 2.0)
                adjusted_score = score
                if modality == "web":
                    content_type = metadata.get("content_type", "")
                    content_quality = metadata.get("content_quality", "normal")
                    url = metadata.get("url", "")
                    if content_quality == "high":
                        adjusted_score *= 1.3
                    elif content_quality == "medium":
                        adjusted_score *= 1.1
                    if content_type == "title":
                        adjusted_score *= 1.8
                    elif content_type == "description":
                        adjusted_score *= 1.4
                    elif content_type == "summary":
                        adjusted_score *= 1.6
                    if metadata.get("embedding_enhanced"):
                        adjusted_score *= 1.2
                    query_lower = query.lower()
                    url_lower = url.lower()
                    title_lower = metadata.get("web_loader_title", "").lower()
                    desc_lower = metadata.get("web_loader_description", "").lower()
                    chunk_lower = metadata.get("chunk", "").lower()
                    if query_lower in url_lower:
                        adjusted_score *= 5.0
                        logger.info(f"Exact domain match found: {query} in {url}")
                    elif metadata.get("domain_boost"):
                        adjusted_score *= metadata.get("domain_boost", 1.0)
                        logger.info(f"Domain boost applied: {metadata.get('domain_boost')}")
                    elif query_lower in title_lower:
                        adjusted_score *= 2.5
                        logger.info(f"Title match found: {query} in title")
                    elif query_lower in desc_lower:
                        adjusted_score *= 2.0
                        logger.info(f"Description match found: {query} in description")
                    elif query_lower in chunk_lower:
                        adjusted_score *= 1.5
                        logger.info(f"Content match found: {query} in chunk")
                raw_results.append({
                    "score": adjusted_score,
                    "original_score": score,
                    "content_quality": metadata.get("content_quality", "normal"),
                    "content_type": metadata.get("content_type", ""),
                    **metadata,
                    "chunk": document
                })
        raw_results.sort(key=lambda x: -x['score'])
        SCORE_GROUP_THRESHOLD = 0.02
        grouped = []
        used = [False] * len(raw_results)
        for i, res in enumerate(raw_results):
            if used[i]:
                continue
            group = [res]
            used[i] = True
            for j in range(i + 1, len(raw_results)):
                if used[j]:
                    continue
                if (abs(res['original_score'] - raw_results[j]['original_score']) < SCORE_GROUP_THRESHOLD and
                    res.get('url') == raw_results[j].get('url')):
                    group.append(raw_results[j])
                    used[j] = True
            group.sort(key=lambda x: -x['score'])
            main = group[0]
            duplicates = group[1:]
            main['duplicates'] = duplicates
            main['has_duplicates'] = len(duplicates) > 0
            grouped.append(main)
        grouped.sort(key=lambda x: -x['score'])
        grouped = grouped[:top_k]
        logger.info(f"Enhanced search for modality '{modality}' returned {len(grouped)} grouped results (top_k={top_k}).")
        return {"results": grouped}
    except Exception as e:
        logger.error(f"Error in enhanced search for modality '{modality}': {e}")
        return {"results": []}

print("🔄 ChromaDB collections initialized...")
indexed_txids = set()
indexed_txids_lock = threading.Lock()
cursor_web = load_cursor("web")
cursor_image = load_cursor("image")
cursor_audio = load_cursor("audio")
cursor_video = load_cursor("video")
print(f"✅ Current cursors: web={cursor_web if cursor_web else 'None'}, image={cursor_image if cursor_image else 'None'}, audio={cursor_audio if cursor_audio else 'None'}, video={cursor_video if cursor_video else 'None'}")

# Initialize locks for thread safety
metas_locks = {m: threading.Lock() for m in MODALITIES}
metas_locks['all'] = threading.Lock()

def store(emb, meta, modality):
    """Store embedding and metadata using ChromaDB."""
    try:
        # Store in main modality collection
        success = store_in_chromadb(emb, meta, modality)
        
        if success:
            # Also store in 'all' collection if not already there
            if modality != "all":
                all_meta = meta.copy()
                all_meta["original_modality"] = modality
                store_in_chromadb(emb, all_meta, "all")
            
            # Add normalized_url to indexed_txids for deduplication
            normalized_url = meta.get("normalized_url", "")
            if normalized_url:
                with indexed_txids_lock:
                    indexed_txids.add(normalized_url)
            
            if modality not in ["image", "web"]:
                logger.info(f"Stored embedding for modality '{modality}' and url '{meta.get('url', '')}'")
        
        return success
        
    except Exception as e:
        logger.error(f"Error storing embedding for {modality}: {e}")
        return False

API_URL = "https://cu.ardrive.io/dry-run?process-id=qNvAoz0TgcH7DMg8BCVn8jF32QH5L6T29VjHxhHqqGE"
HEADERS = {
    'accept': '*/*',
    'content-type': 'application/json',
    'origin': 'https://www.ao.link',
    'referer': 'https://www.ao.link/',
}

def fetch_page(cursor=None, limit=1000, max_retries=5):
    tags = [
        {"name":"Action","value":"Paginated-Records"},
        {"name":"Limit","value":str(limit)},
        {"name":"Sort-By","value":"startTimestamp"},
        {"name":"Sort-Order","value":"desc"},
        {"name":"Data-Protocol","value":"ao"},
        {"name":"Type","value":"Message"},
        {"name":"Variant","value":"ao.TN.1"}
    ]
    if cursor:
        tags.append({"name":"Cursor","value":cursor})
    else:
        tags.append({"name":"Cursor","value":"homedepot"})
    payload = {
        "Id": "1234",
        "Target": "qNvAoz0TgcH7DMg8BCVn8jF32QH5L6T29VjHxhHqqGE",
        "Owner": "1234",
        "Anchor": "0",
        "Data": "1234",
        "Tags": tags
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[ARNS fetch_page] Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    raise Exception(f"[ARNS fetch_page] All {max_retries} attempts failed.")

def get_all_arns():
    all_items = []
    cursor = None
    while True:
        data = fetch_page(cursor)
        message = data.get("Messages", [{}])[0]
        data_str = message.get("Data")
        if not data_str:
            print("No Data field found in response")
            break
        page_data = json.loads(data_str)
        items = page_data.get("items", [])
        all_items.extend(items)
        has_more = page_data.get("hasMore", False)
        cursor = page_data.get("nextCursor")
        print(f"[ARNS] Fetched {len(items)} items, hasMore={has_more}, nextCursor={cursor}")
        if not has_more or not cursor:
            break
    return all_items

def arns_index_loop():
    while True:
        try:
            arns_list = get_all_arns()
            print(f"[ARNS] Total ARNS fetched: {len(arns_list)}")
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for arn in arns_list:
                    url = f"https://{arn['name']}.arweave.net"
                    unique_id = extract_txid_or_arns_name(url)
                    with indexed_txids_lock:
                        already_indexed = unique_id in indexed_txids
                    if not already_indexed:
                        futures.append(executor.submit(index_file, url, True))
                    else:
                        logger.info(f"Skipping already indexed ARNS: {url}")
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ARNS] Error indexing ARNS: {e}")
            time.sleep(600)  # Sleep 10 minutes before next ARNS fetch
        except Exception as e:
            print(f"[ARNS Indexer Error] {e}")
            time.sleep(30)

def index_modality_loop(modality, content_types):
    logger.info(f"Starting index_modality_loop for {modality}")
    cursor = None if modality == "video" else load_cursor(modality)
    while True:
        try:
            query = f'''
            query ($cursor: String) {{
              transactions(
                after: $cursor
                first: {BATCH_SIZE}
                tags: [{{ name: "Content-Type", values: {json.dumps(content_types)} }}]
              ) {{
                pageInfo {{ hasNextPage }}
                edges {{ cursor node {{ id }} }}
              }}
            }}'''
            variables = {"cursor": cursor} if cursor else {}
            data = fetch_graphql(query, variables)
            edges = data["data"]["transactions"]["edges"]
            txid_url_pairs = []
            for edge in edges:
                txid = edge["node"]["id"]
                cursor = edge["cursor"]
                url = f"https://arweave.net/{txid}"
                normalized_url = normalize_url(url)
                with indexed_txids_lock:
                    already_indexed = normalized_url in indexed_txids
                if not already_indexed:
                    detected = detect_file_type(url)
                    if detected == modality:
                        txid_url_pairs.append((normalized_url, url))
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = []
                for normalized_url, url in txid_url_pairs:
                    futures.append(executor.submit(index_file, url))
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Indexing failed: {e}")
            with indexed_txids_lock:
                for normalized_url, _ in txid_url_pairs:
                    indexed_txids.add(normalized_url)
            if modality != "video":
                save_cursor(cursor, modality)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"[{modality.capitalize()} Indexer Error] {e}")
            time.sleep(30)

# --- FastAPI App Initialization (must be before any route decorators) ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    print("🚀 Starting Multimodal Arweave Indexer...")
    logger.info("App startup: launching background indexers.")
   # arns_index_once()  # Run ARNS indexing once at startup
    threading.Thread(target=lambda: index_modality_loop("image", ["image/png", "image/jpeg", "image/webp"]), daemon=True).start()
    threading.Thread(target=lambda: index_modality_loop("audio", ["audio/mpeg", "audio/wav", "audio/mp3"]), daemon=True).start()
    threading.Thread(target=lambda: index_modality_loop("video", ["video/mp4", "video/webm"]), daemon=True).start()
    threading.Thread(target=lambda: index_modality_loop("web", ["application/x.arweave-manifest+json", "text/html"]), daemon=True).start()
    threading.Thread(target=arns_index_loop, daemon=True).start()

class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K

@app.get("/")
def root():
    logger.info("Root endpoint called.")
    return {
        "message": "Multimodal Arweave Search (ChromaDB + ImageBind)", 
        "version": "2.0",
        "features": ["ChromaDB Vector Database", "ImageBind Embeddings", "Multi-modal Search"],
        "status": "running"
    }

@app.get("/status")
def status():
    logger.info("Status endpoint called.")
    status_data = {}
    
    try:
        for modality in MODALITIES:
            count = get_collection_count(modality)
            status_data[modality] = count
            logger.info(f"Status: {modality} has {count} items")
        
        # Add ARNS count
        arns_count = get_collection_count("arns")
        status_data["arns"] = arns_count
        logger.info(f"Status: ARNS has {arns_count} items")
        
        # Add total count
        total_count = sum(status_data.values())
        status_data["total"] = total_count
        logger.info(f"Status: Total items indexed: {total_count}")
            
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        status_data = {m: 0 for m in MODALITIES}
        status_data["arns"] = 0
        status_data["total"] = 0
    
    return status_data

@app.get("/debug/collections")
def debug_collections():
    """Debug endpoint to check ChromaDB collection status"""
    debug_info = {}
    
    for modality in MODALITIES:
        try:
            count = get_collection_count(modality)
            debug_info[modality] = {
                "exists": collections[modality] is not None,
                "count": count,
                "collection_name": f"arweave_{modality}"
            }
        except Exception as e:
            debug_info[modality] = {
                "exists": False,
                "error": str(e)
            }
    
    # Check ARNS
    try:
        arns_count = get_collection_count("arns")
        debug_info["arns"] = {
            "exists": collections["arns"] is not None,
            "count": arns_count,
            "collection_name": "arweave_arns"
        }
    except Exception as e:
        debug_info["arns"] = {
            "exists": False,
            "error": str(e)
        }
    
    return debug_info

def search_modality(query: str, top_k: int, modality: str):
    logger.info(f"Search called for modality '{modality}' with query '{query}' and top_k={top_k}")
    
    # Create query embedding
    query_embedding = embed_text(query)
    
    # Search in ChromaDB
    results = search_in_chromadb(query_embedding, modality, top_k)
    
    # Group results by score (within threshold)
    SCORE_GROUP_THRESHOLD = 0.01
    raw_results = results.get("results", [])
    
    if not raw_results:
        logger.warning(f"No search results for modality '{modality}'")
        return {"results": []}
    
    grouped = []
    used = [False] * len(raw_results)
    for i, res in enumerate(raw_results):
        if used[i]:
            continue
        group = [res]
        used[i] = True
        for j in range(i + 1, len(raw_results)):
            if used[j]:
                continue
            if abs(res['score'] - raw_results[j]['score']) < SCORE_GROUP_THRESHOLD:
                group.append(raw_results[j])
                used[j] = True
        # Sort group by score descending
        group.sort(key=lambda x: -x['score'])
        main = group[0]
        duplicates = group[1:]
        main['duplicates'] = duplicates
        main['has_duplicates'] = len(duplicates) > 0
        grouped.append(main)
    # Sort groups by main score descending and take top_k groups only
    grouped.sort(key=lambda x: -x['score'])
    grouped = grouped[:top_k]
    logger.info(f"Search for modality '{modality}' returned {len(grouped)} grouped results (top_k={top_k}).")
    return {"results": grouped}

# Update ARNS search endpoint to use ARNS-exclusive index/meta
@app.get("/searchweb")
def search_web(query: str, top_k: int = TOP_K):
    """Search web content using ChromaDB"""
    try:
        results = enhanced_search_modality(query, top_k, "web")
        # Unpack if nested
        if isinstance(results, dict) and "results" in results:
            return {"results": results["results"]}
        return {"results": results}
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {"results": []}

@app.get("/searchimage")
def search_image(query: str, top_k: int = TOP_K):
    """Search image content using ChromaDB"""
    try:
        results = retrieve_image(query, top_k)
        if isinstance(results, dict) and "results" in results:
            return {"results": results["results"]}
        return {"results": results}
    except Exception as e:
        logger.error(f"Image search error: {e}")
        return {"results": []}

@app.get("/searchaudio")
def search_audio(query: str, top_k: int = TOP_K):
    """Search audio content using ChromaDB"""
    try:
        results = retrieve_audio(query, top_k)
        if isinstance(results, dict) and "results" in results:
            return {"results": results["results"]}
        return {"results": results}
    except Exception as e:
        logger.error(f"Audio search error: {e}")
        return {"results": []}

@app.get("/searchvideo")
def search_video(query: str, top_k: int = TOP_K):
    """Search video content using ChromaDB"""
    try:
        results = retrieve_video(query, top_k)
        if isinstance(results, dict) and "results" in results:
            return {"results": results["results"]}
        return {"results": results}
    except Exception as e:
        logger.error(f"Video search error: {e}")
        return {"results": []} 

def flatten_result(res):
    # Move all metadata fields to the top level
    meta = res.get('metadata', {})
    flat = {k: v for k, v in meta.items()}
    flat['score'] = res.get('score', 0.0)
    # Use 'document' as 'chunk' if not present
    if 'chunk' not in flat and 'document' in res:
        flat['chunk'] = res['document']
    # Remove unwanted keys
    for k in ['id', 'metadata', 'document']:
        if k in flat:
            del flat[k]
    # Recursively flatten duplicates
    if 'duplicates' in res:
        flat['duplicates'] = [flatten_result(d) for d in res['duplicates']]
    else:
        flat['duplicates'] = []
    flat['has_duplicates'] = bool(flat['duplicates'])
    return flat

def group_results(raw_results, top_k, score_group_threshold=0.01):
    # Convert raw distance to similarity score: score = 1 - distance
    for res in raw_results:
        if 'score' in res:
            res['score'] = 1.0 - res['score']
    grouped = []
    used = [False] * len(raw_results)
    for i, res in enumerate(raw_results):
        if used[i]:
            continue
        group = [res]
        used[i] = True
        for j in range(i + 1, len(raw_results)):
            if used[j]:
                continue
            if abs(res['score'] - raw_results[j]['score']) < score_group_threshold:
                group.append(raw_results[j])
                used[j] = True
        group.sort(key=lambda x: -x['score'])  # Higher similarity = better
        main = group[0]
        duplicates = group[1:]
        main['duplicates'] = duplicates
        main['has_duplicates'] = len(duplicates) > 0
        grouped.append(main)
    grouped.sort(key=lambda x: -x['score'])
    # Flatten all results for frontend compatibility
    return [flatten_result(g) for g in grouped[:top_k]]

# === Retrieval Functions ===
def retrieve_web(prompt, top_k=100):
    """Retrieve web content using ChromaDB, grouped for frontend compatibility"""
    try:
        query_embedding = embed_text(prompt)
        results = search_in_chromadb(query_embedding, "web", top_k * 3)
        raw_results = results.get("results", [])
        grouped = group_results(raw_results, top_k)
        return {"results": grouped}
    except Exception as e:
        logger.error(f"Error in retrieve_web: {e}")
        return {"results": []}

def retrieve_image(prompt, top_k=100):
    """Retrieve image content using ChromaDB, grouped for frontend compatibility"""
    try:
        query_embedding = embed_text(prompt)
        results = search_in_chromadb(query_embedding, "image", top_k * 3)
        raw_results = results.get("results", [])
        grouped = group_results(raw_results, top_k)
        return {"results": grouped}
    except Exception as e:
        logger.error(f"Error in retrieve_image: {e}")
        return {"results": []}

def retrieve_audio(prompt, top_k=100):
    """Retrieve audio content using ChromaDB, grouped for frontend compatibility"""
    try:
        query_embedding = embed_text(prompt)
        results = search_in_chromadb(query_embedding, "audio", top_k * 3)
        raw_results = results.get("results", [])
        grouped = group_results(raw_results, top_k)
        return {"results": grouped}
    except Exception as e:
        logger.error(f"Error in retrieve_audio: {e}")
        return {"results": []}

def retrieve_video(prompt, top_k=100):
    """Retrieve video content using ChromaDB, grouped for frontend compatibility"""
    try:
        query_embedding = embed_text(prompt)
        results = search_in_chromadb(query_embedding, "video", top_k * 3)
        raw_results = results.get("results", [])
        grouped = group_results(raw_results, top_k)
        return {"results": grouped}
    except Exception as e:
        logger.error(f"Error in retrieve_video: {e}")
        return {"results": []} 