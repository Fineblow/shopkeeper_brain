from logging import Logger

from knowledge.processor.import_process.base import BaseNode
from dataclasses import dataclass


# -- 定义数据模型 --
# 未来直接实例化，不用写__init__  __repr__方法
@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str      # 最近的章节标题
    pre_text: str     # 图片上方的正文内容
    post_text: str    # 图片下方的正文内容

@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str                # 图片文件名
    path: str                # 图片完整路径
    context: ImageContext    # 在 MD 中的上下文

# -- 1.文件读写 & 备份 --
class MdFileHandler:
    """
    主要职责：
        1.读取md内容、md_path、图片日志
        2.备份新的md_content（方便测试观看）
    """
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name



# -- 2.图片扫描 & 上下文提取 --
class ImageScanner:
    """
    主要职责：
        1.根据图片目录，得到该目录下的有效的图片文件，目录和非图片排除
        2.去md文件中找到图片的位置
        3.获取该图片的上下文内容，给VLM模型提供图片上下文信息，帮助模型识别图片更加准确
        4.最终组装所有图片的上下文内容(List)
    """
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name


# -- 3.VLM 图片摘要生成 --
class VLMSummarizer:
    """
    主要职责：
        主要根据图片和图片上下文，让VLM生成图片摘要识别
    """
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name


# -- 4.MinIO上传 & 内容替换--
class ImageUploader:
    """
    主要职责：
        1.将本地图片上传到Minio,得到Minio上图片的访问地址
        2.替换 md图片的地址
    """
    def __init__(self, logger:Logger, node_name:str):
        self.logger = logger
        self.node_name = node_name






# --定义主节点--
class MarkDownImageNode(BaseNode):
    """
    编排调度
    """
    name = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state) :
        pass
        # 1.读取文件,如果没有图片路径，直接返回内容

        # 2.扫描图片 & 提取上下文

        # 3.VLM生成摘要

        # 4.上传& 替换

        # 5.备份

        # 6.更新并返回

