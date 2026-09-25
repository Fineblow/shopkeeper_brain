from knowledge.processor.import_process.base import BaseNode


class ImportMilvusNode(BaseNode):
    """
    向量数据入库节点（门面角色）
    协调 Schema 构建、 索引构建  、 数据插入
    """

    name = "import_milvus_node"
    def process(self, state) :
        pass