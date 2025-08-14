from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable, Set, Union

from .store_base import StoreBase, LAZY_ROOT_NAME
from .map_store import MapStore

from ..doc_node import DocNode
from ..index_base import IndexBase
from ..default_index import DefaultIndex
from ..utils import sparse2normal

from lazyllm import LOG
from lazyllm.common import override, obj2str, str2obj
from lazyllm.thirdparty import chromadb


class ChromadbStore(StoreBase):
    """
Inherits from the abstract base class StoreBase. This class is mainly used to store and manage document nodes (DocNode), supporting operations such as node addition, deletion, modification, query, index management, and persistent storage.
Args:
    group_embed_keys (Dict[str, Set[str]]): Specifies the embedding fields associated with each document group.
    embed (Dict[str, Callable]): A dictionary of embedding generation functions, supporting multiple embedding sources.
    embed_dims (Dict[str, int]): The embedding dimensions corresponding to each embedding type.
    dir (str): Path to the chromadb persistent storage directory.
    kwargs (Dict): Additional optional parameters passed to the parent class or internal components.


Examples:
    
    >>> from lazyllm.tools.rag.chroma_store import ChromadbStore
    >>> from typing import Dict, List
    >>> import numpy as np
    >>> store = ChromadbStore(
    ...     group_embed_keys={"articles": {"title_embed", "content_embed"}},
    ...     embed={
    ...         "title_embed": lambda x: np.random.rand(128).tolist(),
    ...         "content_embed": lambda x: np.random.rand(256).tolist()
    ...     },
    ...     embed_dims={"title_embed": 128, "content_embed": 256},
    ...     dir="./chroma_data"
    ... )
    >>> store.update_nodes([node1, node2])
    >>> results = store.query(query_text="文档内容", group_name="articles", top_k=2)
    >>> for node in results:
    ...     print(f"找到文档: {node._content[:20]}...")
    >>> store.remove_nodes(doc_ids=["doc1"])
    """
    def __init__(self, group_embed_keys: Dict[str, Set[str]], embed: Dict[str, Callable],
                 embed_dims: Dict[str, int], dir: str, **kwargs) -> None:
        self._db_client = chromadb.PersistentClient(path=dir)
        LOG.success(f"Initialzed chromadb in path: {dir}")
        node_groups = list(group_embed_keys.keys())
        self._collections: Dict[str, chromadb.api.models.Collection.Collection] = {
            group: self._db_client.get_or_create_collection(group)
            for group in node_groups
        }

        self._map_store = MapStore(node_groups=node_groups, embed=embed)
        self._load_store(embed_dims)

        self._name2index = {
            'default': DefaultIndex(embed, self._map_store),
        }

    @override
    def update_nodes(self, nodes: List[DocNode]) -> None:
        """
Update a group of DocNode objects.
Args:
    nodes (DocNode): The list of DocNode objects to be updated.
"""
        self._map_store.update_nodes(nodes)
        self._save_nodes(nodes)

    @override
    def remove_nodes(self, doc_ids: List[str], group_name: Optional[str] = None,
                     uids: Optional[List[str]] = None) -> None:
        """
Delete nodes based on specified conditions.
Args:
    doc_ids (str): Delete by document ID.
    group_name (str): Specify the group name for deletion.
    uids (str): Delete by unique node ID.
"""
        nodes = self._map_store.get_nodes(group_name=group_name, doc_ids=doc_ids, uids=uids)
        group2uids = defaultdict(list)
        for node in nodes:
            group2uids[node._group].append(node._uid)
        for group, uids in group2uids.items():
            self._delete_group_nodes(group, uids)
            self._map_store.remove_nodes(doc_ids=doc_ids, uids=uids)

    @override
    def update_doc_meta(self, doc_id: str, metadata: dict) -> None:
        """
Update the metadata of a document.
Args:
    doc_id (str): The ID of the document to be updated.
    metadata (dict): The new metadata (key-value pairs).
"""
        self._map_store.update_doc_meta(doc_id=doc_id, metadata=metadata)
        for group in self.activated_groups():
            nodes = self.get_nodes(group_name=group, doc_ids=[doc_id])
            self._save_nodes(nodes)

    @override
    def get_nodes(self, group_name: Optional[str] = None, uids: Optional[List[str]] = None,
                  doc_ids: Optional[Set] = None, **kwargs) -> List[DocNode]:
        """
Query nodes based on specified conditions.
Args:
    group_name (str): The name of the group to which the nodes belong.
    uids (List[str]): A list of unique node IDs.
    doc_ids (Set[str]): A set of document IDs.
    **kwargs: Additional optional parameters.
"""
        return self._map_store.get_nodes(group_name, uids, doc_ids, **kwargs)

    @override
    def activate_group(self, group_names: Union[str, List[str]]) -> bool:
        """
Activate the specified group.
Args:
    group_names([str, List[str]]): Activate by group name.
"""
        return self._map_store.activate_group(group_names)

    @override
    def activated_groups(self):
        """
Activate groups. Return the list of currently activated group names.
"""
        return self._map_store.activated_groups()

    @override
    def is_group_active(self, name: str) -> bool:
        """
Check whether the specified group is active.
Args:
    name (str): The name of the group.
"""
        return self._map_store.is_group_active(name)

    @override
    def all_groups(self) -> List[str]:
        """
Return the list of all group names.
"""
        return self._map_store.all_groups()

    @override
    def query(self, *args, **kwargs) -> List[DocNode]:
        """
Execute a query using the default index.
Args:
    args: Query parameters.
    kwargs: Additional optional parameters.
"""
        return self.get_index('default').query(*args, **kwargs)

    @override
    def register_index(self, type: str, index: IndexBase) -> None:
        """
Register a custom index.
Args:
    type (str): The name of the index type.
    index (IndexBase): An object implementing the IndexBase interface.
"""
        self._name2index[type] = index

    @override
    def get_index(self, type: Optional[str] = None) -> Optional[IndexBase]:
        """
Get the index of the specified type.
Args:
    type (str): The type of the index.
"""
        if type is None:
            type = 'default'
        return self._name2index.get(type)

    @override
    def clear_cache(self, group_names: Optional[List[str]] = None):
        """
Clear the ChromaDB collections and memory cache for specified groups or all groups.
Args:
    group_names (List[str]): List of group names. If None, clear all groups.
"""
        if group_names is None:
            for group_name in self.activated_groups():
                self._db_client.delete_collection(name=group_name)
            self._collections.clear()
            self._map_store.clear_cache()
        elif isinstance(group_names, str):
            group_names = [group_names]
        elif isinstance(group_names, (tuple, list, set)):
            group_names = list(group_names)
        else:
            raise TypeError(f"Invalid type {type(group_names)} for group_names, expected list of str")
        for group_name in group_names:
            self._db_client.delete_collection(name=group_name)
        self._map_store.clear_cache(group_names)

    def _load_store(self, embed_dims: Dict[str, int]) -> None:
        if not self._collections[LAZY_ROOT_NAME].peek(1)["ids"]:
            LOG.info("No persistent data found, skip the rebuilding phrase.")
            return

        # Restore all nodes
        uid2node = {}
        for group in self._collections.keys():
            results = self._peek_all_documents(group)
            nodes = self._build_nodes_from_chroma(results, embed_dims)
            for node in nodes:
                uid2node[node._uid] = node

        # Rebuild relationships
        for node in uid2node.values():
            if node.parent:
                parent_uid = node.parent
                parent_node = uid2node.get(parent_uid)
                node.parent = parent_node
                parent_node.children[node._group].append(node)
        LOG.debug(f"build {group} nodes from chromadb: {nodes}")

        self._map_store.update_nodes(list(uid2node.values()))
        LOG.success("Successfully Built nodes from chromadb.")

    def _save_nodes(self, nodes: List[DocNode]) -> None:
        if not nodes:
            return
        # Note: It's caller's duty to make sure this batch of nodes has the same group.
        group = nodes[0]._group
        ids, embeddings, metadatas, documents = [], [], [], []
        collection = self._collections.get(group)
        assert (
            collection
        ), f"Group {group} is not found in collections {self._collections}"
        for node in nodes:
            metadata = self._make_chroma_metadata(node)
            ids.append(node._uid)
            embeddings.append([0])  # we don't use chroma for retrieving
            metadatas.append(metadata)
            documents.append(obj2str(node._content))
        if ids:
            collection.upsert(
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            LOG.debug(f"Saved {group} nodes {ids} to chromadb.")

    def _delete_group_nodes(self, group_name: str, uids: List[str]) -> None:
        collection = self._collections.get(group_name)
        if collection:
            collection.delete(ids=uids)

    def _build_nodes_from_chroma(self, results: Dict[str, List], embed_dims: Dict[str, int]) -> List[DocNode]:
        nodes: List[DocNode] = []
        for i, uid in enumerate(results['ids']):
            chroma_metadata = results['metadatas'][i]

            parent = chroma_metadata['parent']
            local_metadata = str2obj(chroma_metadata['metadata'])
            global_metadata = str2obj(chroma_metadata['global_metadata']) if not parent else None

            node = DocNode(
                uid=uid,
                content=str2obj(results["documents"][i]),
                group=chroma_metadata["group"],
                embedding=str2obj(chroma_metadata['embedding']),
                parent=parent,
                metadata=local_metadata,
                global_metadata=global_metadata,
            )

            if node.embedding:
                # convert sparse embedding to List[float]
                new_embedding_dict = {}
                for key, embedding in node.embedding.items():
                    if isinstance(embedding, dict):
                        dim = embed_dims.get(key)
                        if not dim:
                            raise ValueError(f'dim of embed [{key}] is not determined.')
                        new_embedding_dict[key] = sparse2normal(embedding, dim)
                    else:
                        new_embedding_dict[key] = embedding
                node.embedding = new_embedding_dict

            nodes.append(node)
        return nodes

    def _make_chroma_metadata(self, node: DocNode) -> Dict[str, Any]:
        metadata = {
            "group": node._group,
            "parent": node.parent._uid if node.parent else "",
            "embedding": obj2str(node.embedding),
            "metadata": obj2str(node._metadata),
        }

        if node.is_root_node:
            metadata["global_metadata"] = obj2str(node.global_metadata)

        return metadata

    def _peek_all_documents(self, group: str) -> Dict[str, List]:
        assert group in self._collections, f"group {group} not found."
        collection = self._collections[group]
        return collection.peek(collection.count())
