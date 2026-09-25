import threading
from typing import Optional
import logging

from minio import Minio
from pymilvus import MilvusClient
from dotenv import load_dotenv
from knowledge.utils.client.base import BaseClientManager

logger = logging.getLogger(__name__)
load_dotenv()


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    @classmethod
    def get_minio(cls) -> Minio:
        return cls._get_or_create(
            "_minio_client", cls._minio_lock, cls._create_minio
        )

    @classmethod
    def _create_minio(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(
                endpoint, access_key=access_key,
                secret_key=secret_key, secure=False
            )

            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
            else:
                logger.info(f"MinIO bucket '{bucket_name}' 已存在")

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise  # 配置缺失，直接往上抛
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e