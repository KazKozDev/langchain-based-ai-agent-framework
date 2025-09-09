"""
Vector store implementation for RAG system.
Supports multiple vector databases: ChromaDB, FAISS, etc.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Vector store manager for document embeddings and similarity search.
    """
    
    def __init__(
        self,
        store_type: str = "chroma",
        persist_directory: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "langchain_agent_docs"
    ):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of vector store ("chroma", "faiss")
            persist_directory: Directory to persist the vector store
            embedding_model: HuggingFace embedding model name
            collection_name: Name of the collection/index
        """
        self.logger = logging.getLogger(__name__)
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        
        # Setup persistence directory
        if persist_directory is None:
            # Use project root directory for vector store
            project_root = Path(__file__).parent.parent.parent
            persist_directory = os.path.join(project_root, "data", "vector_store")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def _initialize_vector_store(self):
        """Initialize the vector store based on type."""
        try:
            if self.store_type == "chroma":
                try:
                    self._init_chroma()
                except Exception as e:
                    self.logger.warning(f"Chroma initialization failed ('{e}'). Falling back to FAISS...")
                    self.store_type = "faiss"
                    self._init_faiss()
            elif self.store_type == "faiss":
                self._init_faiss()
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
            self.logger.info(f"Vector store initialized: {self.store_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _init_chroma(self):
        """Initialize ChromaDB vector store."""
        persist_dir = self.persist_directory / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        persist_path = str(persist_dir)
        
        try:
            # Try to load or create store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_path
            )
            self.logger.info("ChromaDB collection ready")
        except Exception as e:
            msg = str(e)
            self.logger.warning(f"Chroma init error: {msg}")
            # Handle schema mismatch (e.g., 'no such column: collections.topic') by resetting the dir
            if "no such column" in msg or "OperationalError" in msg or "schema" in msg:
                try:
                    import shutil
                    self.logger.warning("Detected Chroma schema mismatch. Resetting persisted directory...")
                    if persist_dir.exists():
                        shutil.rmtree(persist_dir)
                    persist_dir.mkdir(parents=True, exist_ok=True)
                    self.vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(persist_dir)
                    )
                    self.logger.info("ChromaDB collection recreated after schema reset")
                except Exception as e2:
                    self.logger.error(f"Failed to reset Chroma persistence dir: {e2}")
                    raise
            else:
                # Unknown error; re-raise
                raise
    
    def _init_faiss(self):
        """Initialize FAISS vector store."""
        faiss_path = self.persist_directory / "faiss_index"
        faiss_path.mkdir(parents=True, exist_ok=True)
        
        if faiss_path.exists():
            try:
                # Load existing FAISS index
                self.vector_store = FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Existing FAISS index loaded")
            except Exception as e:
                self.logger.warning(f"Failed to load existing FAISS index: {e}")
                self.vector_store = None
        
        if self.vector_store is None:
            # Create new FAISS index with a dummy document
            dummy_doc = Document(
                page_content="Dummy document for initialization",
                metadata={"source": "initialization", "doc_id": "dummy"}
            )
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            self.logger.info("New FAISS index created")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        try:
            if not documents:
                self.logger.warning("No documents provided")
                return []
            
            # Split documents into chunks
            chunked_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': f"{doc.metadata.get('doc_id', uuid.uuid4().hex)}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'parent_doc_id': doc.metadata.get('doc_id', uuid.uuid4().hex)
                    })
                    chunked_docs.append(chunk)
            
            # Add to vector store
            if self.store_type == "chroma":
                ids = self.vector_store.add_documents(chunked_docs)
            elif self.store_type == "faiss":
                ids = self.vector_store.add_documents(chunked_docs)
                # Persist FAISS index
                self.vector_store.save_local(str(self.persist_directory / "faiss_index"))
            
            self.logger.info(f"Added {len(chunked_docs)} document chunks to vector store")
            return ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of similar documents
        """
        try:
            if self.vector_store is None:
                self.logger.warning("Vector store not initialized")
                return []
            
            # Perform search based on store type
            if self.store_type == "chroma":
                if filter_metadata:
                    results = self.vector_store.similarity_search(
                        query=query,
                        k=k,
                        filter=filter_metadata
                    )
                else:
                    results = self.vector_store.similarity_search(query=query, k=k)
            elif self.store_type == "faiss":
                results = self.vector_store.similarity_search(query=query, k=k)
                # Apply manual filtering for FAISS if needed
                if filter_metadata:
                    results = [
                        doc for doc in results 
                        if all(
                            doc.metadata.get(key) == value 
                            for key, value in filter_metadata.items()
                        )
                    ]
            
            self.logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {e}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if self.vector_store is None:
                self.logger.warning("Vector store not initialized")
                return []
            
            if self.store_type == "chroma":
                if filter_metadata:
                    results = self.vector_store.similarity_search_with_score(
                        query=query,
                        k=k,
                        filter=filter_metadata
                    )
                else:
                    results = self.vector_store.similarity_search_with_score(query=query, k=k)
            elif self.store_type == "faiss":
                results = self.vector_store.similarity_search_with_score(query=query, k=k)
                # Apply manual filtering for FAISS if needed
                if filter_metadata:
                    results = [
                        (doc, score) for doc, score in results 
                        if all(
                            doc.metadata.get(key) == value 
                            for key, value in filter_metadata.items()
                        )
                    ]
            
            self.logger.info(f"Found {len(results)} scored results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing scored similarity search: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.store_type == "chroma":
                self.vector_store.delete(doc_ids)
            elif self.store_type == "faiss":
                self.logger.warning("FAISS does not support document deletion by ID")
                return False
            
            self.logger.info(f"Deleted {len(doc_ids)} documents from vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dictionary with collection info
        """
        try:
            info = {
                "store_type": self.store_type,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embeddings.model_name
            }
            
            if self.store_type == "chroma":
                try:
                    collection = self.vector_store._collection
                    info["document_count"] = collection.count()
                except:
                    info["document_count"] = "Unknown"
            elif self.store_type == "faiss":
                try:
                    info["document_count"] = self.vector_store.index.ntotal
                except:
                    info["document_count"] = "Unknown"
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.store_type == "chroma":
                # Delete the collection and recreate
                self.vector_store.delete_collection()
                self._init_chroma()
            elif self.store_type == "faiss":
                # Delete the FAISS index files
                faiss_path = self.persist_directory / "faiss_index"
                if faiss_path.exists():
                    import shutil
                    shutil.rmtree(faiss_path)
                self._init_faiss()
            
            self.logger.info("Vector store collection cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False