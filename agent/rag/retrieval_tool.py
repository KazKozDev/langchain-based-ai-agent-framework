"""
RAG retrieval tool for the LangChain agent.
Provides document retrieval capabilities using vector similarity search.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_core.tools import Tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

from .vector_store import VectorStore
from .document_processor import DocumentProcessor


class RAGRetrievalTool:
    """
    RAG tool for document retrieval and question answering.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        store_type: str = "faiss",
        persist_directory: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "langchain_agent_docs"
    ):
        """
        Initialize RAG retrieval tool.
        
        Args:
            vector_store: Pre-initialized vector store (optional)
            store_type: Type of vector store to create if not provided
            persist_directory: Directory to persist vector store
            embedding_model: Embedding model name
            collection_name: Collection name for vector store
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        if vector_store is None:
            self.vector_store = VectorStore(
                store_type=store_type,
                persist_directory=persist_directory,
                embedding_model=embedding_model,
                collection_name=collection_name
            )
        else:
            self.vector_store = vector_store
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
    
    class RAGRetrievalInput(BaseModel):
        """Input schema for RAG retrieval tool."""
        query: str = Field(description="Search query for the knowledge base")
        k: int = Field(default=5, description="Number of documents to retrieve (1-20)")
        with_scores: bool = Field(default=False, description="Include relevance scores in results")
        filter_metadata: Optional[Dict[str, str]] = Field(default=None, description="Metadata filters as key-value pairs")
        
    def get_tool(self) -> StructuredTool:
        """
        Get LangChain StructuredTool instance for RAG retrieval.
        
        Returns:
            StructuredTool instance
        """
        return StructuredTool(
            name="rag_retrieval",
            description="Retrieve relevant documents from the knowledge base using semantic search",
            args_schema=self.RAGRetrievalInput,
            func=self._retrieve_documents_structured
        )
    
    def _retrieve_documents_structured(
        self,
        query: str,
        k: int = 5,
        with_scores: bool = False,
        filter_metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Retrieve documents using structured input.
        
        Args:
            query: Search query
            k: Number of results to return
            with_scores: Include relevance scores
            filter_metadata: Metadata filters
            
        Returns:
            Formatted retrieval results
        """
        try:
            # Perform retrieval
            if with_scores:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter_metadata=filter_metadata
                )
                return self._format_scored_results(results, query)
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter_metadata=filter_metadata
                )
                return self._format_results(results, query)
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _retrieve_documents(self, input_str: str) -> str:
        """
        Retrieve documents based on input query.
        
        Args:
            input_str: Input string with query and optional parameters
            
        Returns:
            Formatted retrieval results
        """
        try:
            # Parse input
            params = self._parse_input(input_str)
            query = params.get('query', input_str.strip())
            k = int(params.get('k', 5))
            with_scores = params.get('with_scores', 'false').lower() == 'true'
            filter_str = params.get('filter')
            
            # Parse filter
            filter_metadata = None
            if filter_str:
                filter_metadata = self._parse_filter(filter_str)
            
            # Perform retrieval
            if with_scores:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter_metadata=filter_metadata
                )
                return self._format_scored_results(results, query)
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter_metadata=filter_metadata
                )
                return self._format_results(results, query)
            
        except Exception as e:
            error_msg = f"Error retrieving documents: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _parse_input(self, input_str: str) -> Dict[str, str]:
        """
        Parse input string to extract parameters.
        
        Args:
            input_str: Input string
            
        Returns:
            Dictionary of parsed parameters
        """
        params = {}
        
        if '|' in input_str:
            parts = input_str.split('|')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    params[key.strip()] = value.strip()
                else:
                    # First part without ':' is the query
                    if 'query' not in params:
                        params['query'] = part.strip()
        else:
            params['query'] = input_str.strip()
        
        return params
    
    def _parse_filter(self, filter_str: str) -> Dict[str, str]:
        """
        Parse filter string into metadata dictionary.
        
        Args:
            filter_str: Filter string in format "key=value,key2=value2"
            
        Returns:
            Dictionary of filter metadata
        """
        filter_metadata = {}
        
        if filter_str:
            pairs = filter_str.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    filter_metadata[key.strip()] = value.strip()
        
        return filter_metadata
    
    def _format_results(self, results: List[Document], query: str) -> str:
        """
        Format retrieval results for display.
        
        Args:
            results: List of retrieved documents
            query: Original query
            
        Returns:
            Formatted results string
        """
        if not results:
            return f"No documents found for query: '{query}'"
        
        output = [f"Found {len(results)} relevant documents for query: '{query}'\n"]
        
        for i, doc in enumerate(results, 1):
            output.append(f"--- Document {i} ---")
            
            # Add metadata info
            metadata = doc.metadata
            if 'source' in metadata:
                output.append(f"Source: {metadata['source']}")
            if 'filename' in metadata:
                output.append(f"File: {metadata['filename']}")
            if 'chunk_index' in metadata:
                output.append(f"Chunk: {metadata['chunk_index']}/{metadata.get('total_chunks', '?')}")
            
            # Add content preview
            content = doc.page_content.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            
            output.append(f"Content: {content}")
            output.append("")  # Empty line
        
        return "\n".join(output)
    
    def _format_scored_results(self, results: List[tuple], query: str) -> str:
        """
        Format scored retrieval results for display.
        
        Args:
            results: List of (document, score) tuples
            query: Original query
            
        Returns:
            Formatted results string
        """
        if not results:
            return f"No documents found for query: '{query}'"
        
        output = [f"Found {len(results)} relevant documents for query: '{query}' (with relevance scores)\n"]
        
        for i, (doc, score) in enumerate(results, 1):
            output.append(f"--- Document {i} (Score: {score:.4f}) ---")
            
            # Add metadata info
            metadata = doc.metadata
            if 'source' in metadata:
                output.append(f"Source: {metadata['source']}")
            if 'filename' in metadata:
                output.append(f"File: {metadata['filename']}")
            if 'chunk_index' in metadata:
                output.append(f"Chunk: {metadata['chunk_index']}/{metadata.get('total_chunks', '?')}")
            
            # Add content preview
            content = doc.page_content.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            
            output.append(f"Content: {content}")
            output.append("")  # Empty line
        
        return "\n".join(output)
    
    def add_documents_from_files(self, file_paths: List[str]) -> str:
        """
        Add documents from file paths to the vector store.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Status message
        """
        try:
            all_documents = []
            
            for file_path in file_paths:
                documents = self.document_processor.process_file(file_path)
                all_documents.extend(documents)
            
            if all_documents:
                # Validate documents
                valid_docs = self.document_processor.validate_documents(all_documents)
                
                # Add to vector store
                doc_ids = self.vector_store.add_documents(valid_docs)
                
                return f"Successfully added {len(valid_docs)} documents from {len(file_paths)} files to knowledge base."
            else:
                return "No valid documents found in the provided files."
            
        except Exception as e:
            error_msg = f"Error adding documents from files: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def add_documents_from_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Add documents from a directory to the vector store.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to search recursively
            file_patterns: List of file patterns to include
            
        Returns:
            Status message
        """
        try:
            documents = self.document_processor.process_directory(
                directory_path=directory_path,
                recursive=recursive,
                file_patterns=file_patterns
            )
            
            if documents:
                # Validate documents
                valid_docs = self.document_processor.validate_documents(documents)
                
                # Add to vector store
                doc_ids = self.vector_store.add_documents(valid_docs)
                
                return f"Successfully added {len(valid_docs)} documents from directory '{directory_path}' to knowledge base."
            else:
                return f"No valid documents found in directory '{directory_path}'."
            
        except Exception as e:
            error_msg = f"Error adding documents from directory: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def add_text_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a text document directly to the vector store.
        
        Args:
            text: Text content
            metadata: Optional metadata
            
        Returns:
            Status message
        """
        try:
            document = self.document_processor.process_text(text, metadata)
            doc_ids = self.vector_store.add_documents([document])
            
            return f"Successfully added text document to knowledge base (ID: {document.metadata.get('doc_id', 'unknown')})."
            
        except Exception as e:
            error_msg = f"Error adding text document: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_collection_info(self) -> str:
        """
        Get information about the vector store collection.
        
        Returns:
            Collection information
        """
        try:
            info = self.vector_store.get_collection_info()
            
            output = ["Vector Store Information:"]
            for key, value in info.items():
                output.append(f"- {key.replace('_', ' ').title()}: {value}")
            
            return "\n".join(output)
            
        except Exception as e:
            error_msg = f"Error getting collection info: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def clear_knowledge_base(self) -> str:
        """
        Clear all documents from the knowledge base.
        
        Returns:
            Status message
        """
        try:
            success = self.vector_store.clear_collection()
            
            if success:
                return "Successfully cleared all documents from knowledge base."
            else:
                return "Failed to clear knowledge base."
                
        except Exception as e:
            error_msg = f"Error clearing knowledge base: {e}"
            self.logger.error(error_msg)
            return error_msg


class RAGManagementTool:
    """
    Tool for managing RAG system - adding documents, clearing collection, etc.
    """
    
    def __init__(self, rag_tool: RAGRetrievalTool):
        """
        Initialize RAG management tool.
        
        Args:
            rag_tool: RAG retrieval tool instance
        """
        self.rag_tool = rag_tool
        self.logger = logging.getLogger(__name__)
    
    class RAGManagementInput(BaseModel):
        """Input schema for RAG management tool."""
        action: str = Field(description="Action to perform: add_file, add_files, add_directory, add_text, info, clear")
        path: Optional[str] = Field(default=None, description="File or directory path for add operations")
        content: Optional[str] = Field(default=None, description="Text content for add_text action")
        title: Optional[str] = Field(default=None, description="Title for text documents")
        recursive: bool = Field(default=True, description="Search recursively in directories")
        patterns: Optional[list] = Field(default=None, description="File patterns to include (e.g., ['*.py', '*.md'])")
    
    def get_tool(self) -> StructuredTool:
        """
        Get LangChain StructuredTool instance for RAG management.
        
        Returns:
            StructuredTool instance
        """
        return StructuredTool(
            name="rag_management",
            description="Manage the RAG knowledge base - add documents, clear collection, get info",
            args_schema=self.RAGManagementInput,
            func=self._manage_rag_structured
        )
    
    def _manage_rag_structured(
        self,
        action: str,
        path: Optional[str] = None,
        content: Optional[str] = None,
        title: Optional[str] = None,
        recursive: bool = True,
        patterns: Optional[list] = None
    ) -> str:
        """
        Manage RAG system using structured input.
        
        Args:
            action: Action to perform
            path: File or directory path
            content: Text content for add_text
            title: Title for text documents
            recursive: Search recursively in directories
            patterns: File patterns to include
            
        Returns:
            Operation result
        """
        try:
            action = action.lower().strip()
            
            if action == 'info':
                return self.rag_tool.get_collection_info()
            
            elif action == 'clear':
                return self.rag_tool.clear_knowledge_base()
            
            elif action == 'add_file':
                if not path:
                    return "Error: path parameter required for add_file action"
                return self.rag_tool.add_documents_from_files([path])
            
            elif action == 'add_files':
                if not path:
                    return "Error: path parameter required for add_files action"
                file_paths = [f.strip() for f in path.split(',')]
                return self.rag_tool.add_documents_from_files(file_paths)
            
            elif action == 'add_directory':
                if not path:
                    return "Error: path parameter required for add_directory action"
                return self.rag_tool.add_documents_from_directory(
                    directory_path=path,
                    recursive=recursive,
                    file_patterns=patterns
                )
            
            elif action == 'add_text':
                if not content:
                    return "Error: content parameter required for add_text action"
                metadata = {}
                if title:
                    metadata['title'] = title
                return self.rag_tool.add_text_document(content, metadata)
            
            else:
                return f"Unknown action: {action}. Available actions: add_file, add_files, add_directory, add_text, info, clear"
            
        except Exception as e:
            error_msg = f"Error managing RAG system: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _manage_rag(self, input_str: str) -> str:
        """
        Manage RAG system based on input.
        
        Args:
            input_str: Input string with action and parameters
            
        Returns:
            Operation result
        """
        try:
            # Parse action and parameters
            if ':' not in input_str:
                if input_str.strip().lower() == 'info':
                    return self.rag_tool.get_collection_info()
                elif input_str.strip().lower() == 'clear':
                    return self.rag_tool.clear_knowledge_base()
                else:
                    return "Invalid action. Use 'info' or 'clear', or specify action:parameters format."
            
            action, params_str = input_str.split(':', 1)
            action = action.strip().lower()
            
            if action == 'add_file':
                file_path = params_str.strip()
                return self.rag_tool.add_documents_from_files([file_path])
            
            elif action == 'add_files':
                file_paths = [f.strip() for f in params_str.split(',')]
                return self.rag_tool.add_documents_from_files(file_paths)
            
            elif action == 'add_directory':
                # Parse directory parameters
                params = self._parse_directory_params(params_str)
                return self.rag_tool.add_documents_from_directory(
                    directory_path=params['directory'],
                    recursive=params.get('recursive', True),
                    file_patterns=params.get('patterns')
                )
            
            elif action == 'add_text':
                # Parse text parameters
                params = self._parse_text_params(params_str)
                metadata = {}
                if 'title' in params:
                    metadata['title'] = params['title']
                return self.rag_tool.add_text_document(params['text'], metadata)
            
            else:
                return f"Unknown action: {action}. Available actions: add_file, add_files, add_directory, add_text, info, clear"
            
        except Exception as e:
            error_msg = f"Error managing RAG system: {e}"
            self.logger.error(error_msg)
            return error_msg
    
    def _parse_directory_params(self, params_str: str) -> Dict[str, Any]:
        """Parse directory action parameters."""
        params = {'directory': params_str.strip()}
        
        if '|' in params_str:
            parts = params_str.split('|')
            params['directory'] = parts[0].strip()
            
            for part in parts[1:]:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'recursive':
                        params['recursive'] = value.lower() == 'true'
                    elif key == 'patterns':
                        params['patterns'] = [p.strip() for p in value.split(',')]
        
        return params
    
    def _parse_text_params(self, params_str: str) -> Dict[str, str]:
        """Parse text action parameters."""
        params = {'text': params_str.strip()}
        
        if '|' in params_str:
            parts = params_str.split('|')
            params['text'] = parts[0].strip()
            
            for part in parts[1:]:
                if ':' in part:
                    key, value = part.split(':', 1)
                    params[key.strip()] = value.strip()
        
        return params