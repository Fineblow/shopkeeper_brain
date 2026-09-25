from knowledge.processor.import_process.base import BaseNode


class BgeEmbeddingChunksNode(BaseNode):
    """
    主要职责：
    1.获取所有chunks拼接要向量的内容
    2.批量嵌入chunk的 (embedding_content :  item_name + chunk.get('content'))
    3.将所有chunk嵌入后的向量值，存储到列表中，在返回给下一个节点用
    """

    name = "bge_embedding_chunks_node"

    def process(self, state):
        pass