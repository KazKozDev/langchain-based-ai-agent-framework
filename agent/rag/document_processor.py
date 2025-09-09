"""
Document processing utilities for RAG system.
Handles various document formats: PDF, DOCX, TXT, MD, etc.
"""

import logging
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import hashlib

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class DocumentProcessor:
    """
    Document processor for ingesting various document formats into RAG system.
    """
    
    def __init__(self):
        """Initialize document processor."""
        self.logger = logging.getLogger(__name__)
        
        # Supported file types
        self.supported_extensions = {
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.py': self._load_text,
            '.js': self._load_text,
            '.json': self._load_text,
            '.yaml': self._load_text,
            '.yml': self._load_text,
            '.xml': self._load_text,
            '.html': self._load_text,
            '.css': self._load_text,
            '.sql': self._load_text,
        }
        
        # Add PDF support if available
        if PDF_AVAILABLE:
            self.supported_extensions['.pdf'] = self._load_pdf
        
        # Add DOCX support if available
        if DOCX_AVAILABLE:
            self.supported_extensions['.docx'] = self._load_docx
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if supported, False otherwise
        """
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return list(self.supported_extensions.keys())
    
    def process_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Process a single file into Document objects.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return []
            
            if not self.is_supported(str(file_path)):
                self.logger.warning(f"Unsupported file type: {file_path.suffix}")
                return []
            
            # Generate document ID
            doc_id = self._generate_doc_id(str(file_path))
            
            # Prepare metadata
            doc_metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'file_extension': file_path.suffix,
                'doc_id': doc_id,
                'file_size': file_path.stat().st_size,
                'created_at': file_path.stat().st_ctime,
                'modified_at': file_path.stat().st_mtime,
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Load document using appropriate loader
            loader_func = self.supported_extensions[file_path.suffix.lower()]
            documents = loader_func(str(file_path))
            
            # Update metadata for all documents
            for doc in documents:
                doc.metadata.update(doc_metadata)
            
            self.logger.info(f"Processed file: {file_path.name} -> {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search recursively
            file_patterns: List of file patterns to include (e.g., ['*.py', '*.md'])
            exclude_patterns: List of patterns to exclude
            
        Returns:
            List of Document objects
        """
        try:
            directory = Path(directory_path)
            
            if not directory.exists() or not directory.is_dir():
                self.logger.error(f"Directory not found: {directory_path}")
                return []
            
            all_documents = []
            
            # Get all files
            if recursive:
                files = directory.rglob('*')
            else:
                files = directory.glob('*')
            
            # Filter files
            files = [f for f in files if f.is_file()]
            
            # Apply patterns
            if file_patterns:
                filtered_files = []
                for pattern in file_patterns:
                    filtered_files.extend(directory.rglob(pattern) if recursive else directory.glob(pattern))
                files = [f for f in files if f in filtered_files]
            
            # Exclude patterns
            if exclude_patterns:
                for pattern in exclude_patterns:
                    exclude_files = list(directory.rglob(pattern) if recursive else directory.glob(pattern))
                    files = [f for f in files if f not in exclude_files]
            
            # Process each file
            for file_path in files:
                if self.is_supported(str(file_path)):
                    documents = self.process_file(str(file_path))
                    all_documents.extend(documents)
            
            self.logger.info(f"Processed directory: {directory_path} -> {len(all_documents)} documents")
            return all_documents
            
        except Exception as e:
            self.logger.error(f"Error processing directory {directory_path}: {e}")
            return []
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Process raw text into a Document object.
        
        Args:
            text: Raw text content
            metadata: Metadata for the document
            
        Returns:
            Document object
        """
        doc_id = self._generate_doc_id(text)
        
        doc_metadata = {
            'source': 'text_input',
            'doc_id': doc_id,
            'content_length': len(text),
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        return Document(
            page_content=text,
            metadata=doc_metadata
        )
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text file."""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load markdown file."""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            return loader.load()
        except Exception as e:
            # Fallback to text loader
            self.logger.warning(f"Markdown loader failed for {file_path}, using text loader: {e}")
            return self._load_text(file_path)
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            self.logger.error(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def _load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX file."""
        try:
            loader = Docx2txtLoader(file_path)
            return loader.load()
        except Exception as e:
            self.logger.error(f"Error loading DOCX file {file_path}: {e}")
            return []
    
    def _generate_doc_id(self, content: str) -> str:
        """
        Generate unique document ID based on content.
        
        Args:
            content: Content to generate ID from
            
        Returns:
            Unique document ID
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def extract_metadata_from_content(self, content: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from document content.
        
        Args:
            content: Document content
            file_path: Optional file path
            
        Returns:
            Extracted metadata
        """
        metadata = {
            'content_length': len(content),
            'word_count': len(content.split()),
            'line_count': content.count('\n') + 1,
        }
        
        if file_path:
            # Try to extract title from content
            lines = content.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and not line.startswith('#'):
                    # Use first non-empty, non-comment line as potential title
                    if len(line) < 100:  # Reasonable title length
                        metadata['extracted_title'] = line
                    break
                elif line.startswith('# '):
                    # Markdown title
                    metadata['extracted_title'] = line[2:].strip()
                    break
        
        return metadata
    
    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Validate and clean document list.
        
        Args:
            documents: List of documents to validate
            
        Returns:
            List of valid documents
        """
        valid_docs = []
        
        for doc in documents:
            # Check if document has content
            if not doc.page_content or not doc.page_content.strip():
                self.logger.warning("Skipping empty document")
                continue
            
            # Ensure metadata exists
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            
            # Add doc_id if missing
            if 'doc_id' not in doc.metadata:
                doc.metadata['doc_id'] = self._generate_doc_id(doc.page_content)
            
            valid_docs.append(doc)
        
        self.logger.info(f"Validated {len(valid_docs)}/{len(documents)} documents")
        return valid_docs