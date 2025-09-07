#!/usr/bin/env python3
"""
Configuration management for LINE RAG Bot
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion

load_dotenv()

class Config:
    """System configuration"""
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
    PINECONE_INDEX_NAME = "cafe-line-bot"
    EMBEDDING_DIMENSION = 384
    OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"
    
    @classmethod
    def validate_config(cls):
        """Validate required environment variables"""
        missing = []
        if not cls.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.LINE_CHANNEL_ACCESS_TOKEN:
            missing.append("LINE_CHANNEL_ACCESS_TOKEN")
        if not cls.LINE_CHANNEL_SECRET:
            missing.append("LINE_CHANNEL_SECRET")
            
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        return True