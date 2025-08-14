from typing import List, Optional
from .doc_node import DocNode
from .index_base import IndexBase
from lazyllm.common import override
from .store import MapStore, MilvusStore

class SmartEmbeddingIndex(IndexBase):
    """Smart embedding index class for managing and querying document node embeddings.

SmartEmbeddingIndex is an index implementation based on IndexBase that supports multiple backend storage options (such as Milvus, Map storage). It provides a unified interface for updating, removing, and querying document nodes, particularly suitable for RAG (Retrieval-Augmented Generation) applications that require efficient vector similarity search.

This class encapsulates the implementation details of different storage backends, providing users with a clean and consistent API interface.

Args:
    backend_type (str): Backend storage type. Supported types include:
        - 'milvus': Uses Milvus vector database as backend storage
        - 'map': Uses memory-mapped storage as backend storage
    **kwargs: Additional parameters passed to the specific backend storage, which may vary depending on the backend_type.

**Raises:**

- ValueError: Raised when an unsupported backend_type is provided
"""
    def __init__(self, backend_type: str, **kwargs):
        if backend_type == 'milvus':
            self._store = MilvusStore(**kwargs)
        elif backend_type == 'map':
            self._store = MapStore(**kwargs)
        else:
            raise ValueError(f'unsupported backend [{backend_type}]')

    @override
    def update(self, nodes: List[DocNode]) -> None:
        """Update document nodes in the index.

This method adds a list of new document nodes to the index, including their embedding vectors and metadata. If nodes already exist, their content will be updated.

Args:
    nodes (List[DocNode]): List of document nodes to add or update. Each DocNode should contain text content, embedding vectors, and related metadata.
"""
        self._store.update_nodes(nodes)

    @override
    def remove(self, uids: List[str], group_name: Optional[str] = None) -> None:
        """Remove specified document nodes from the index.

This method removes corresponding document nodes from the index based on the provided list of unique identifiers.

Args:
    uids (List[str]): List of unique identifiers of document nodes to be removed.
    group_name (Optional[str]): Node group name used to specify removal of nodes from a specific group. Defaults to None, meaning all matching nodes will be removed.
"""
        self._store.remove_nodes(uids=uids)

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        """Query the most relevant document nodes in the index.

This method returns a list of document nodes most relevant to the query based on vector similarity search. The specific query parameters and behavior depend on the implementation of the underlying storage backend.

Args:
    *args: Positional arguments passed to the underlying storage's query method.
    **kwargs: Keyword arguments passed to the underlying storage's query method. 

**Returns:**

- List[DocNode]: List of document nodes sorted by relevance.
"""
        return self._store.query(*args, **kwargs)
