from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """倒序融合排序"""
    name = "rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        pass