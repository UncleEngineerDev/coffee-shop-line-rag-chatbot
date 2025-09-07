#!/usr/bin/env python3
"""
Setup Script for Coffee Corner RAG Bot
Creates Pinecone index and imports cafe data
"""

import json
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from sentence_transformers import SentenceTransformer
from config import Config

def setup_pinecone_index():
    """Create Pinecone index"""
    print("📊 Setting up Pinecone index...")
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if Config.PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            )
        )
        print(f"✅ Created index: {Config.PINECONE_INDEX_NAME}")
    else:
        print(f"✅ Index already exists: {Config.PINECONE_INDEX_NAME}")

def import_cafe_data():
    """Import cafe data to Pinecone"""
    print("🏪 Importing cafe data...")
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    with open('cafe_data.json', 'r', encoding='utf-8') as f:
        cafe_docs = json.load(f)
    
    vectors = []
    for i, doc in enumerate(cafe_docs):
        text = f"{doc['title']} {doc['content']}"
        vector = embedder.encode(text).tolist()
        
        vectors.append({
            'id': f'cafe_{i}',
            'values': vector,
            'metadata': {
                'title': doc['title'],
                'content': doc['content'],
                'source_url': doc.get('source_url', ''),
                'type': doc.get('type', '')
            }
        })
    index.upsert(vectors)
    print(f"✅ Uploaded {len(vectors)} documents")
    
    return len(vectors)

def verify_setup():
    """Verify setup completion"""
    print("🔍 Verifying setup...")
    
    from app import SimpleRAGBot
    
    bot = SimpleRAGBot()
    test_queries = ["เมนู", "ราคา", "เวลาเปิด"]
    
    for query in test_queries:
        results = bot.search_knowledge(query, top_k=1)
        if results:
            print(f"✅ Query '{query}': Found '{results[0]['title']}'")
        else:
            print(f"❌ Query '{query}': No results")
    
    print("\n🤖 Testing full RAG...")
    result = bot.process_message("ราคาลาเต้เท่าไร?")
    print(f"Response: {result['reply'][:100]}...")
    
    return True

if __name__ == "__main__":
    print("🚀 Coffee Corner RAG Bot Setup")
    print("=" * 50)
    
    try:
        setup_pinecone_index()
        count = import_cafe_data()
        verify_setup()
        
        print(f"\n🎉 Setup completed! {count} documents ready")
        print("📱 Run: python app.py (for LINE bot)")
        print("🧪 Run: python app.py test (for testing)")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("💡 Check your .env file and API keys")