"""
Memory Store Module
Long-term memory using ChromaDB, optimized for low-RAM systems.

DESIGN DECISIONS:
- ChromaDB for lightweight vector storage (vs heavier FAISS)
- Explicit memory tags: temporary, project, permanent
- User can inspect, edit, delete memory
- Only store relevant project info, not trivial facts
- Efficient retrieval for M4 systems
"""

import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

# Lazy import for chromadb
_chromadb = None


def get_chromadb():
    """Lazy import chromadb to speed up startup"""
    global _chromadb
    if _chromadb is None:
        try:
            import chromadb
            _chromadb = chromadb
        except ImportError:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )
    return _chromadb


class MemoryTag(str, Enum):
    """Memory persistence tags"""
    TEMPORARY = "temporary"  # Session-only, cleared on restart
    PROJECT = "project"      # Project-specific, persisted
    PERMANENT = "permanent"  # User preferences, always persisted


class MemoryEntry(BaseModel):
    """A single memory entry"""
    id: str
    content: str
    tag: MemoryTag
    category: str  # e.g., "environment", "preference", "constraint", "goal"
    created_at: str
    updated_at: str
    metadata: dict = {}


class MemoryStore:
    """
    Long-term memory storage using ChromaDB.
    
    Features:
    - Store project constraints, environment info, user preferences
    - Retrieve relevant memories for context injection
    - User can inspect, edit, delete memories
    - Efficient for low-RAM M4 systems
    
    Memory Categories:
    - environment: OS, architecture, tools, versions
    - preference: User preferences, style choices
    - constraint: Project constraints, requirements
    - goal: Ongoing goals, objectives
    - fact: Verified facts learned during conversation
    """
    
    def __init__(
        self,
        persist_directory: str = "./memory_store",
        collection_name: str = "agent_memory"
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    def _ensure_initialized(self):
        """Initialize ChromaDB if not already done"""
        if self._client is not None:
            return
        
        chromadb = get_chromadb()
        
        # Create persist directory if needed
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Use persistent client for data retention
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
    
    def _generate_id(self) -> str:
        """Generate unique ID for memory entry"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _now(self) -> str:
        """Current timestamp"""
        return datetime.utcnow().isoformat()
    
    def add(
        self,
        content: str,
        tag: MemoryTag = MemoryTag.PROJECT,
        category: str = "fact",
        metadata: Optional[dict] = None
    ) -> MemoryEntry:
        """
        Add a new memory entry.
        
        Args:
            content: The memory content
            tag: Persistence tag (temporary, project, permanent)
            category: Category for organization
            metadata: Additional metadata
            
        Returns:
            The created MemoryEntry
        """
        self._ensure_initialized()
        
        memory_id = self._generate_id()
        now = self._now()
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            tag=tag,
            category=category,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        # Store in ChromaDB
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "tag": tag.value,
                "category": category,
                "created_at": now,
                "updated_at": now,
                **(metadata or {})
            }]
        )
        
        return entry
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        tag_filter: Optional[MemoryTag] = None,
        category_filter: Optional[str] = None,
        similarity_threshold: float = 0.7
    ) -> list[MemoryEntry]:
        """
        Query memories by semantic similarity.
        
        Args:
            query_text: Text to search for
            n_results: Maximum results to return
            tag_filter: Filter by tag
            category_filter: Filter by category
            similarity_threshold: Minimum similarity (0-1)
            
        Returns:
            List of matching MemoryEntry objects
        """
        self._ensure_initialized()
        
        # Build where clause for filters
        where = {}
        if tag_filter:
            where["tag"] = tag_filter.value
        if category_filter:
            where["category"] = category_filter
        
        # Query ChromaDB
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )
        
        entries = []
        if results and results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                # Convert distance to similarity (cosine distance -> similarity)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    entries.append(MemoryEntry(
                        id=memory_id,
                        content=results["documents"][0][i],
                        tag=MemoryTag(metadata.get("tag", "project")),
                        category=metadata.get("category", "fact"),
                        created_at=metadata.get("created_at", ""),
                        updated_at=metadata.get("updated_at", ""),
                        metadata={
                            k: v for k, v in metadata.items()
                            if k not in ["tag", "category", "created_at", "updated_at"]
                        }
                    ))
        
        return entries
    
    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        self._ensure_initialized()
        
        result = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"]
        )
        
        if result and result["ids"]:
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            return MemoryEntry(
                id=memory_id,
                content=result["documents"][0],
                tag=MemoryTag(metadata.get("tag", "project")),
                category=metadata.get("category", "fact"),
                created_at=metadata.get("created_at", ""),
                updated_at=metadata.get("updated_at", ""),
                metadata={
                    k: v for k, v in metadata.items()
                    if k not in ["tag", "category", "created_at", "updated_at"]
                }
            )
        
        return None
    
    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tag: Optional[MemoryTag] = None,
        category: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[MemoryEntry]:
        """
        Update an existing memory entry.
        
        Args:
            memory_id: ID of memory to update
            content: New content (or None to keep existing)
            tag: New tag (or None to keep existing)
            category: New category (or None to keep existing)
            metadata: New metadata to merge
            
        Returns:
            Updated MemoryEntry or None if not found
        """
        self._ensure_initialized()
        
        existing = self.get(memory_id)
        if not existing:
            return None
        
        # Merge updates
        new_content = content if content is not None else existing.content
        new_tag = tag if tag is not None else existing.tag
        new_category = category if category is not None else existing.category
        new_metadata = {**existing.metadata, **(metadata or {})}
        now = self._now()
        
        # Update in ChromaDB
        self._collection.update(
            ids=[memory_id],
            documents=[new_content],
            metadatas=[{
                "tag": new_tag.value,
                "category": new_category,
                "created_at": existing.created_at,
                "updated_at": now,
                **new_metadata
            }]
        )
        
        return MemoryEntry(
            id=memory_id,
            content=new_content,
            tag=new_tag,
            category=new_category,
            created_at=existing.created_at,
            updated_at=now,
            metadata=new_metadata
        )
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        self._ensure_initialized()
        
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    
    def list_all(
        self,
        tag_filter: Optional[MemoryTag] = None,
        category_filter: Optional[str] = None
    ) -> list[MemoryEntry]:
        """
        List all memories, optionally filtered.
        
        Args:
            tag_filter: Filter by tag
            category_filter: Filter by category
            
        Returns:
            List of all matching memories
        """
        self._ensure_initialized()
        
        # Build where clause
        where = {}
        if tag_filter:
            where["tag"] = tag_filter.value
        if category_filter:
            where["category"] = category_filter
        
        # Get all from collection
        result = self._collection.get(
            where=where if where else None,
            include=["documents", "metadatas"]
        )
        
        entries = []
        if result and result["ids"]:
            for i, memory_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                entries.append(MemoryEntry(
                    id=memory_id,
                    content=result["documents"][i],
                    tag=MemoryTag(metadata.get("tag", "project")),
                    category=metadata.get("category", "fact"),
                    created_at=metadata.get("created_at", ""),
                    updated_at=metadata.get("updated_at", ""),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ["tag", "category", "created_at", "updated_at"]
                    }
                ))
        
        return entries
    
    def clear_temporary(self):
        """Clear all temporary memories (session cleanup)"""
        self._ensure_initialized()
        
        # Get all temporary memories
        result = self._collection.get(
            where={"tag": MemoryTag.TEMPORARY.value},
            include=[]
        )
        
        if result and result["ids"]:
            self._collection.delete(ids=result["ids"])
    
    def get_context_for_query(
        self,
        query: str,
        max_memories: int = 5
    ) -> str:
        """
        Get relevant context from memory for a query.
        Returns formatted string for injection into LLM prompt.
        
        Args:
            query: The user's query
            max_memories: Maximum memories to include
            
        Returns:
            Formatted context string
        """
        memories = self.query(query, n_results=max_memories)
        
        if not memories:
            return ""
        
        context_parts = ["Relevant context from memory:"]
        for mem in memories:
            context_parts.append(f"- [{mem.category}] {mem.content}")
        
        return "\n".join(context_parts)


# Singleton instance
_store: Optional[MemoryStore] = None


def get_memory_store(
    persist_directory: str = "./memory_store",
    collection_name: str = "agent_memory"
) -> MemoryStore:
    """Get or create singleton MemoryStore instance"""
    global _store
    if _store is None:
        _store = MemoryStore(persist_directory, collection_name)
    return _store
