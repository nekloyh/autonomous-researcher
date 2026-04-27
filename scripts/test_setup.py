"""Verify all services are working."""
import os

from dotenv import load_dotenv

load_dotenv()

def test_groq():
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    r = llm.invoke("Say OK")
    print(f"✅ Groq: {r.content[:50]}")

def test_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    r = llm.invoke("Say OK")
    print(f"✅ Gemini: {r.content[:50]}")

def test_ollama_embed():
    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(model="nomic-embed-text")
    v = emb.embed_query("hello")
    print(f"✅ Ollama embed: dim={len(v)}")

def test_qdrant():
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    collections = client.get_collections()
    print(f"✅ Qdrant: {len(collections.collections)} collections")

def test_tavily():
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    r = client.search("LangChain", max_results=1)
    print(f"✅ Tavily: {len(r['results'])} results")

def test_langsmith():
    from langsmith import Client
    client = Client()
    client.list_runs(project_name=os.getenv("LANGSMITH_PROJECT"), limit=1)
    print("✅ LangSmith: connected")

if __name__ == "__main__":
    test_groq()
    test_gemini()
    test_ollama_embed()
    test_qdrant()
    test_tavily()
    test_langsmith()
    print("\n🎉 All services ready!")
