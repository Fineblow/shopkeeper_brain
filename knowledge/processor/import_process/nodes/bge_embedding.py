import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    """
    主要职责：
    1.获取所有chunks拼接要向量的内容
    2.批量嵌入chunk的 (embedding_content :  item_name + chunk.get('content'))
    3.将所有chunk嵌入后的向量值，存储到列表中，在返回给下一个节点用
    """

    name = "bge_embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        validate_chunks,config = self._validate_get_inputs(state)

        # 2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config,'embedding_batch_chunk',16)

        # 3. 准备分批嵌入
        total_length = len(validate_chunks)
        final_chunks = []
        for i in range(0, total_length, embedding_batch_chunk_size):
            batch = validate_chunks[i:i+embedding_batch_chunk_size]
            batch_chunks = self._process_batch_chunks(batch,i,total_length)
            final_chunks.extend(batch_chunks)

        # 4. 更新&返回state
        state['chunks'] = final_chunks
        return state

    def _validate_get_inputs(self, state):
        config = get_config()
        self.log_step("step1","参数校验")
        chunks = state.get('chunks')
        if not chunks or not isinstance(chunks,list):
            raise ValidationError(f"chunks为空或无效",self.name)
        return chunks,config

    def _process_batch_chunks(self, batch,index,total_length):
        """处理一个批次的切片"""
        self.log_step("step2", f"开始批量处理chunk嵌入:批次{index + 1}-{index + len(batch)}")

        # 1. 循环处理所有 chunk 的要嵌入的内容拼接
        embedding_contents = []
        for _,chunk in enumerate(batch):
            item_name = chunk.get('item_name')
            content = chunk.get('content')
            embedding_content =f"{item_name}\n{content}"
            embedding_contents.append(embedding_content)

        # 2. 批量嵌入
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            embedding_result = bge_m3_client.encode_documents(embedding_contents)

            if not embedding_result:
                self.logger.warning(f"嵌入后的结果不存在...")
                return batch
        except Exception as e:
            self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
            return batch

        # 3. 循环处理所有 chunk 的向量以及注入到每一个 chunk 中
        for index,chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构 csr 矩阵 & 获取稀疏向量
            start_index = embedding_result['sparse'].indptr[index]
            end_index = embedding_result['sparse'].indptr[index+1]
            token_id = embedding_result['sparse'].indices[start_index:end_index].tolist()
            weight = embedding_result['sparse'].data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            # 3.3 注入
            chunk['dense_vector'] = dense_vector
            chunk['sparse_vector'] = sparse_vector
        self.logger.info(f"完成批量处理chunk嵌入:批次{index + 1}-{index + len(batch)}/{total_length}")
        return batch

if __name__ == "__main__":
    setup_logging()

    base_temp_dir = Path(r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto")
    input_path = base_temp_dir/"chunks_item_name.json"
    output_path = base_temp_dir/"chunks_vector.json"

    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state = {
        "chunks": content
    }

    node = BgeEmbeddingChunksNode()

    result = node(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")