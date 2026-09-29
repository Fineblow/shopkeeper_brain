import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段规范"""
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 预定义的标量字段（复用）
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)

class _MilvusSchemaBuilder:
    @staticmethod
    def build(milvus_client, dim):
        logger.info("开始构建约束(schema)...")
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        for scalar_field in _SCALAR_FIELDS:
            kwargs = {"field_name": scalar_field.field_name, "datatype": scalar_field.datatype}
            if scalar_field.max_length is not None:
                kwargs["max_length"] = scalar_field.max_length
            schema.add_field(**kwargs)

        logger.info(f"构建约束(schema)完成...")
        return schema

class _MilvusIndexBuilder:
    @staticmethod
    def build(milvus_client, collection_name):
        logger.info("开始构建索引(index)...")
        index = milvus_client.prepare_index_params(collection_name=collection_name)
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        logger.info(f"构建索引(index)完成...")
        return index

class _MilvusInserter:

    def __init__(self, client, collection_name):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks):
        logger.info(f"开始插入{len(chunks)}块到Milvus...")
        inserted_result = self._client.insert(collection_name=self._collection_name, data=chunks)
        inserted_count = inserted_result.get('insert_count')
        ids = inserted_result.get('ids')

        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}记录,并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks, ids):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id


class ImportMilvusNode(BaseNode):
    """
    向量数据入库节点（门面角色）
    协调 Schema 构建、 索引构建  、 数据插入
    """

    name = "import_milvus_node"
    def process(self, state) :
        # 1.参数校验
        validate_chunks,config,dim = self._validate_get_input(state)

        # 2.获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()

        if not milvus_client:
            return state

        # 3.获取集合名字
        collection = getattr(config,"chunks_collection")

        # 4.检查集合
        self._ensure_has_collection(milvus_client, collection,dim)

        # 5. 插入
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        final_chunks = inserter.insert(chunks=validate_chunks)

        # 6. 更新state
        state['chunks'] = final_chunks

        return state

    def _validate_get_input(self, state):
        self.log_step("step1","参数校验")
        config = get_config()
        chunks = state.get("chunks")
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        validate_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validate_chunks.append(chunk)
            else:
                self.logger.error("切块chunk缺少向量数据", self.name)

        if not validate_chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        dim = len(validate_chunks[0].get('dense_vector'))
        self.logger.info(f"导入Milvus向量数据库的有效块：{len(validate_chunks)},且chunk的向量维度{dim}")
        return validate_chunks,config,dim

    def _ensure_has_collection(self, milvus_client, collection_name, dim,delete_flag=False):
        self.log_step("step2","检查集合")
        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"删除集合：{collection_name}")
            milvus_client.drop_collection(collection_name=collection_name)

        if milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"集合：{collection_name}已存在")
            return

        schema = _MilvusSchemaBuilder.build(milvus_client, dim)
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)

if __name__ == "__main__":
    setup_logging()

    temp_dir = Path(r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

        state: ImportGraphState = {
            "chunks": content.get("chunks", [])
        }

        import_milvus = ImportMilvusNode()
        result_state = import_milvus(state)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result_state, fh, ensure_ascii=False, indent=4)

            logger.info(f"备份临时文件{output_path}成功")


