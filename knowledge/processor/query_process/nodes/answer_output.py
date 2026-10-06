from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class AnswerOutputNode(BaseNode):
    """生成答案解决"""

    name = "answer_output"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        pass