from knowledge.processor.import_process.base import BaseNode


class DocumentSplitNode(BaseNode):
    """
    文档切分节点
    处理流程：
        1.读md内容, 参数校验, 换行符统计
        2.按标题切割
    """

    name = "document_split_node"
    def process(self, state) :
        pass