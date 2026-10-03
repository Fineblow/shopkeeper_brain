import datetime
import logging
import os.path
import shutil
import uuid


from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import kb_import_graph_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_utils import add_running_task, add_done_task, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

logger = logging.getLogger(__name__)

class ImportFileService:
    def _get_date_path(self)-> str:
        return os.path.join(get_local_base_dir(), datetime.datetime.now().strftime("%Y%m%d"))

    def upload_file(self,file:UploadFile):
        """
        处理上传文件：
        1. 生成 task_id，构建归档目录
        2. 标记 upload_file 节点为运行中
        3. 保存文件到本地磁盘
        4. 同步上传到 MinIO
        5. 标记 upload_file 节点完成
        6. 返回 (task_id, file_dir, import_file_path)
        """
        # 1.生成task_id
        task_id = str(uuid.uuid4().hex[0:8]) # 真正的随机 获取前8个随机数
        # 2.文件上传存储路径
        date_path = self._get_date_path()
        file_dir = os.path.join(date_path, task_id)
        # 不在创建，在忽略
        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id, "upload_file")# 名称与task_util.py中 英文转中文名称一致。

        # 3.双写
        # 3.1 保存到本地
        import_file_path = self.upload_file_to_local(file, file_dir)

        # 3.2 保存到minio     file.name用于minio存储子路径名称
        self.upload_file_to_minio(import_file_path,file.filename)

        add_done_task(task_id, "upload_file")

        return task_id, file_dir, import_file_path

    def upload_file_to_local(self, file, file_dir):
        """将上传文件保存到本地"""
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        try:
            import_file_path = os.path.join(file_dir, file.filename)
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file,f)    # 带缓冲区写操作。效率高   1Mb
        except Exception as e:
            logger.error(f"文件保存本地出错: {e}")
            raise FileProcessingError(f"文件保存本地出错: {e}")
        return import_file_path


    def upload_file_to_minio(self, import_file_path, file_name):
        """将上传文件保存到minio"""
        # 1.获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"获取minio客户端出错: {e}")
            return
        try:
            config = get_config()
            obj_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}"
            minio_client.fput_object(config.minio_bucket, obj_name, import_file_path)
        except Exception as e:
            logger.warning(f"文件保存minio出错: {e}")
            return

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """
        启动导入流程 langgraph
        """
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }
        update_task_status(task_id,TASK_STATUS_PROCESSING)
        try:
            for event in kb_import_graph_app.stream(init_state):
                for node_name,process_state in event.items():
                    print(f"{task_id}-运行节点: {node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.info(f"导入流程执行出错: {e}")

