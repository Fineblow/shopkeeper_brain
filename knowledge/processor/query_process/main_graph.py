from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search import HyDeSearchNode
from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.rerank import RerankNode
from knowledge.processor.query_process.nodes.rrf import RrfNode
from knowledge.processor.query_process.nodes.vector_search import VectorSearchNode
from knowledge.processor.query_process.nodes.web_search_mcp import WebSearchMcpNode
from knowledge.processor.query_process.state import QueryGraphState

load_dotenv()

def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。
    """
    if state.get('answer'):
        return True
    else:
        return False


def create_query_graph() -> CompiledStateGraph:
    """
    创建查询流程图。
    """
    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)
    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,  # 虚拟节点（分发）
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp": WebSearchMcpNode(),
        "join": lambda x: {},  # 虚拟节点（汇合）
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode()
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            True: "answer_output",
            False: "multi_search"
        }
    )

    # 6. 多路搜索分发（并行执行）
    workflow.add_edge("multi_search","search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding","join")
    workflow.add_edge("search_embedding_hyde","join")
    workflow.add_edge("web_search_mcp","join")

    # 8. 顺序边
    workflow.add_edge( "join","rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge( "rerank","answer_output")
    workflow.add_edge("answer_output",END)

    # 9. 返回可运行的状态
    return workflow.compile()

# 创建全局图实例
query_app = create_query_graph()

if __name__ == "__main__":
    setup_logging()
    print("=" * 60)
    print("开始测试: 查询流程主图 (main_graph)")
    print("=" * 60)

    # 测试场景：商品名明确，走完整 pipeline
    print("\n【场景】: 商品名明确，走完整 pipeline")
    print("-" * 60)

    mock_state_1 = {
        "original_query": "RS-12 数字万用表如何测量直流电压？",
        "session_id": "test_session_main_graph",
        "task_id": "test_task_001",
        "is_stream": True,
    }

    print(f"  查询: {mock_state_1['original_query']}")
    print(f"  session_id: {mock_state_1['session_id']}")

    result_1 = query_app.invoke(mock_state_1)  # 只能看到最后结果
    # result_1 = query_app.stream(mock_state_1) #可以查看到每一个节点的输出
    # 与LLM中的llm.invoke()和 llm.stream()的作用不同。llm.stream()表示流式输出。

    print(f"\n  【结果】:")
    print(f"  商品名: {result_1.get('item_names')}")
    print(f"  重写查询: {result_1.get('rewritten_query')}")
    answer_1 = result_1.get("answer", "")
    print(f"  答案: {answer_1[:200]}..." if len(answer_1) > 200 else f"  答案: {answer_1}")

    query_app.get_graph().print_ascii()
