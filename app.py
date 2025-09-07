#!/usr/bin/env python3
"""
Coffee Corner LINE RAG Bot
RAG + LINE Bot + OpenRouter integration
"""

from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests
import json
from typing import Dict, List
from config import Config

Config.validate_config()
app = Flask(__name__)

line_bot_api = LineBotApi(Config.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(Config.LINE_CHANNEL_SECRET)


class SimpleRAGBot:
    """Simple RAG Bot combining search and generation"""
    
    def __init__(self):
        print("🤖 เริ่มต้น Simple RAG Bot...")
        
        self.pinecone = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pinecone.Index(Config.PINECONE_INDEX_NAME)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ RAG Bot พร้อมใช้งาน!")
    
    def search_knowledge(self, query: str, top_k: int = 4) -> List[Dict]:
        """Search for relevant documents in Pinecone"""
        query_vector = self.embedder.encode(query).tolist()
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        documents = []
        for match in results['matches']:
            documents.append({
                'title': match['metadata'].get('title', ''),
                'content': match['metadata'].get('content', ''),
                'score': match['score']
            })
        
        return documents
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using context documents and LLM"""
        
        context = "\n".join([
            f"📄 {doc['title']}: {doc['content']}"
            for doc in context_docs
        ])
        prompt = f"""คุณคือผู้ช่วยของร้านกาแฟ "Coffee Corner" 🏪

กฎการตอบ:
- ตอบเป็นภาษาไทยและใส่อีโมจิ
- ใช้ข้อมูลที่ให้มาเท่านั้น
- ตอบสั้น เหมาะสำหรับ LINE chat
- ใส่ราคาชัดเจน
- ไม่ตอบเป็น markdown format

ข้อมูลร้าน:
{context}

คำถาม: {query}
ตอบ:"""

        return self._call_openrouter(prompt)
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API for text generation"""
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={
                    "model": Config.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.3
                },
                headers={
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"OpenRouter Error: {response.status_code}")
                return "ขออภัยค่ะ ระบบมีปัญหา กรุณาลองใหม่ค่ะ 🙏"
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return "ขออภัยค่ะ ไม่สามารถตอบได้ กรุณาติดต่อร้านค่ะ 📞"
    
    def process_message(self, user_message: str) -> Dict:
        """Complete RAG pipeline: RETRIEVE → AUGMENT → GENERATE"""
        try:
            documents = self.search_knowledge(user_message)
            
            if not documents:
                return {
                    "reply": "ขออภัยค่ะ ไม่พบข้อมูล กรุณาติดต่อร้าน LINE @coffeecorner ค่ะ 📞",
                    "quick_replies": ["☕ เมนูเครื่องดื่ม", "🕐 เวลาเปิด-ปิด"]
                }
            
            answer = self.generate_response(user_message, documents)
            if not answer.strip():
                answer = "ขออภัยค่ะ ไม่สามารถตอบได้ กรุณาติดต่อร้านค่ะ 📞"
            
            return {
                "reply": answer,
                "quick_replies": ["☕ เมนูเครื่องดื่ม", "🍕 เมนูอาหาร", "🕐 เวลาเปิด-ปิด", "📍 ที่อยู่"],
                "sources": [doc['title'] for doc in documents[:2]]
            }
            
        except Exception as e:
            print(f"RAG Error: {e}")
            return {
                "reply": "ขออภัยค่ะ เกิดข้อผิดพลาด กรุณาลองใหม่ค่ะ 🙏",
                "quick_replies": ["☕ เมนู", "📞 ติดต่อร้าน"]
            }

rag_bot = SimpleRAGBot()


@app.route("/webhook", methods=['POST'])
def webhook():
    """Handle LINE Platform webhook"""
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
        return jsonify({"status": "OK"})
    except InvalidSignatureError:
        return jsonify({"error": "Invalid signature"}), 400

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """Process incoming LINE messages"""
    
    user_message = event.message.text
    user_id = event.source.user_id
    
    print(f"📱 Received: {user_message}")
    
    rag_result = rag_bot.process_message(user_message)
    reply_text = rag_result['reply']
    quick_replies = None
    if rag_result.get('quick_replies'):
        quick_reply_items = [
            QuickReplyButton(action=MessageAction(label=label, text=label))
            for label in rag_result['quick_replies']
        ]
        quick_replies = QuickReply(items=quick_reply_items)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text, quick_reply=quick_replies)
    )
    
    print(f"🤖 Replied: {reply_text[:50]}...")

@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Coffee Corner RAG Bot is running! ☕"
    })


def test_rag():
    """Test RAG functionality"""
    print("🧪 Testing RAG Bot...")
    
    test_queries = [
        "สวัสดีครับ",
        "ราคาลาเต้เท่าไร?",
        "ร้านเปิดกี่โมง?",
        "มีที่จอดรถไหม?"
    ]
    
    for query in test_queries:
        print(f"\n👤 {query}")
        result = rag_bot.process_message(query)
        print(f"🤖 {result['reply']}")
        print(f"📚 Sources: {result.get('sources', [])}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rag()
    else:
        app.run(host='0.0.0.0', port=8000, debug=True)