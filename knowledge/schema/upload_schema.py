"""
使用 Pydantic 定义 API 的数据契约
"""
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应"""
    message: str = Field(...,description="响应消息")
    task_id: str = Field(...,description="任务ID")

class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    status:str = Field(...,description="任务状态")
    done_list: list[str] = Field(...,description="已完成节点列表")
    running_list: list[str] = Field(...,description="正在运行节点列表")
    durations: dict[str,float] = Field(default={},description="各节点耗时(秒)")

