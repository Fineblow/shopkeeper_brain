import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith.schemas import DataType

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    """商品名识别节点"""
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建商品名识别上下文
        item_name_recognition_context = self._prepare_item_name_recognition_context(
            chunks, item_name_chunks_k, item_name_chunk_size
        )

        # 3. LLM商品名识别
        item_name = self._recognition_name(file_title, item_name_recognition_context)

        # 4. 向量化提取到商品名
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 存储到milvus中
        self._insert_milvus(file_title, item_name, dense_vector, sparse_vector,
                            self.config.item_name_collection)

        # 6. 回填item_name信息
        self._fill_item_name(item_name, state, chunks)

        # 7.备份，给下个节点准备下测试数据。
        self._backup_chunks(state, chunks)

        return state

    def _validate_state(self, state):
        self.log_step("step1","参数校验")

        file_title = state.get('file_title')
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)

        chunks = state.get('chunks')
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(self.name, "chunks", list)

        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <=0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        return file_title,chunks,item_name_chunk_k,item_name_chunk_size

    def _prepare_item_name_recognition_context(self, chunks, item_name_chunks_k, item_name_chunk_size):
        self.log_step("step2", "构建商品名识别上下文")
        total = 0
        final_context = []

        for idx,chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk, dict):
                continue
            chunk_content = chunk.get('content')
            context = f"【切片】-{idx}-{chunk_content}"

            if total + len(context) > item_name_chunk_size:
                break
            total+=len(context)
            final_context.append(context)

        return '\n'.join(final_context)

    def _recognition_name(self, file_title, item_name_recognition_context):
        self.log_step("step3", "LLM商品名识别")
        try:
            llm_client = AIClients.get_llm_openai(response_format = False)
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_recognition_context)

            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            llm_result = llm_response.content.strip()
            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                return file_title

            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return json.loads(llm_result).get("item_name")

        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name):
        self.log_step("step4", "商品名向量化提取")
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])

            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))
            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None

    def _insert_milvus(self, file_title, item_name, dense_vector, sparse_vector, item_name_collection):
        if not dense_vector or not sparse_vector:
            self.logger.info(f"商品名 [{item_name}] 向量化处理失败，跳过插入Milvus")
            return
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端获取失败: {e}")
            return
        try:
            if not milvus_client.has_collection(item_name_collection):
                self._creat_item_name_collection(item_name_collection,milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _creat_item_name_collection(self, item_name_collection, milvus_client):
        schema = milvus_client.create_schema()
        schema.add_field(field_name='pk', datatype=DataType.VARCHAR, is_primary=True, auto_id=True,max_legth=100)
        schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector",index_name="dense_vector_index",
                              index_type="AUTOINDEX",metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector",index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX",metric_type="IP")
        milvus_client.create_collection(collection_name=item_name_collection, schema=schema, index_params=index_param)
        self.logger.info(f"集合 {item_name_collection} 创建成功并构建了索引")

    def _fill_item_name(self, item_name, state, chunks):
        for chunk in chunks:
            chunk['item_name'] = item_name
        state['item_name']=item_name

    def _backup_chunks(self, state, chunks):
        """
             将回填item_name的切片列表结果备份到json文件，给下个节点单元测试使用。
             :param state:
             :param chunks:
             :return:
             """
        local_dir = state.get("file_dir", "")
        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")

if __name__ =="__main__":
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "hak180产品安全手册",
        "file_dir": r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node(state)

    # # 5. 输出结果
    # print(f"商品名: {result.get('item_name')}")
    # print(f"chunks数量: {len(result.get('chunks', []))}")
    # print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")
    from rich import print as uprint

    uprint(result)