from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import re
from pathlib import Path
from typing import List, Dict


def extract_card_metadata(card_text: str, category: str) -> Dict[str, str]:
    """
    Extract metadata from a credit card chunk.
    
    Args:
        card_text: The text content of the card chunk
        category: The category (derived from filename, e.g., 'travel')
    
    Returns:
        Dictionary containing metadata fields
    """
    metadata = {
        "category": category,
        "card_name": "",
        "bank": "",
        "reward_type": "",
        "use_case": category  # Default use_case to category
    }
    
    # Extract card name (first line with number pattern like "1. Card Name")
    card_name_match = re.search(r'^\d+\.\s*(.+?)(?:\n|$)', card_text, re.MULTILINE)
    if card_name_match:
        metadata["card_name"] = card_name_match.group(1).strip()
    
    # Extract bank name from card name
    card_name_lower = metadata["card_name"].lower()
    if "axis" in card_name_lower:
        metadata["bank"] = "Axis Bank"
    elif "hsbc" in card_name_lower:
        metadata["bank"] = "HSBC"
    elif "hdfc" in card_name_lower:
        metadata["bank"] = "HDFC Bank"
    elif "sbi" in card_name_lower:
        metadata["bank"] = "SBI"
    elif "icici" in card_name_lower:
        metadata["bank"] = "ICICI Bank"
    # Add more banks as needed
    
    # Determine reward type based on content
    card_text_lower = card_text.lower()
    if "miles" in card_text_lower or "edge miles" in card_text_lower:
        metadata["reward_type"] = "miles"
    elif "cashback" in card_text_lower:
        metadata["reward_type"] = "cashback"
    elif "reward points" in card_text_lower or "points" in card_text_lower:
        metadata["reward_type"] = "points"
    else:
        metadata["reward_type"] = "rewards"  # Generic fallback
    
    return metadata


def chunk_by_credit_card(documents: List[Document]) -> List[Document]:
    """
    Chunk documents by credit card entries.
    Each credit card becomes one self-contained chunk.
    
    Strategy:
    - Split on numbered card entries (e.g., "1. Card Name", "2. Card Name")
    - Keep all sections together: name, fees, features, rewards, lounge, conclusion
    - Attach metadata: category, card_name, bank, reward_type, use_case
    
    Args:
        documents: List of loaded documents from DirectoryLoader
    
    Returns:
        List of Document objects, one per credit card
    """
    chunked_documents = []
    
    for doc in documents:
        # Extract category from filename (e.g., "travel.md" -> "travel")
        source_path = Path(doc.metadata.get("source", ""))
        category = source_path.stem  # Gets filename without extension
        
        content = doc.page_content
        
        # Split by numbered card pattern (e.g., "1. ", "2. ", "3. ")
        # This regex looks for lines starting with a number followed by a period
        card_pattern = r'(?=^\d+\.\s+[A-Z])'
        
        # Split the content into card chunks
        card_chunks = re.split(card_pattern, content, flags=re.MULTILINE)
        
        # Filter out empty chunks and process each card
        for chunk in card_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Extract metadata for this card
            card_metadata = extract_card_metadata(chunk, category)
            
            # Create a new Document with the chunk and metadata
            chunked_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,  # Preserve original metadata (like source)
                    **card_metadata  # Add our custom metadata
                }
            )
            
            chunked_documents.append(chunked_doc)
    
    return chunked_documents


# Load documents from the data folder
loader = DirectoryLoader(
    './data/', 
    glob='*.md', 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=False
)

docs = loader.load()

# Apply custom chunking strategy: 1 credit card = 1 chunk
chunked_docs = chunk_by_credit_card(docs)

# Display results
print(f"\n{'='*80}")
print(f"CHUNKING RESULTS")
print(f"{'='*80}\n")
print(f"Original documents loaded: {len(docs)}")
print(f"Total chunks created: {len(chunked_docs)}\n")

for i, chunk in enumerate(chunked_docs, 1):
    print(f"\n{'-'*80}")
    print(f"CHUNK {i}")
    print(f"{'-'*80}")
    print(f"Card Name: {chunk.metadata.get('card_name', 'N/A')}")
    print(f"Bank: {chunk.metadata.get('bank', 'N/A')}")
    print(f"Category: {chunk.metadata.get('category', 'N/A')}")
    print(f"Reward Type: {chunk.metadata.get('reward_type', 'N/A')}")
    print(f"Use Case: {chunk.metadata.get('use_case', 'N/A')}")
    print(f"Source: {chunk.metadata.get('source', 'N/A')}")
    print(f"\nContent Preview (first 200 chars):")
    print(f"{chunk.page_content[:200]}...")
    print(f"\nContent Length: {len(chunk.page_content)} characters")
    # Rough token estimate (1 token ≈ 4 characters)
    estimated_tokens = len(chunk.page_content) // 4
    print(f"Estimated Tokens: ~{estimated_tokens}")
