import json
from pathlib import Path

from knowledge.processor.import_process import state
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import FileProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    入口节点：
    根据输入文件的扩展名设置相应的处理标志。
    决定后续流程走PDF转换分支  还是直接处理MD分支
    """

    name = "entry"

    def process(self,state:ImportGraphState)-> ImportGraphState:
        """
                处理文件类型的检测
                :param state: ImportGraphState该节点处理之前的节点状态
                :return: ImportGraphState： 该节点处理之后的节点状态
                """
        # 1.获取导入文件的路径以及文件所在的目录
        self.log_step("step1","[获取文件路径]")
        file_dir = state.get("file_dir")
        import_file_path = state.get("import_file_path")

        # 2.检查文件路径是否为空
        self.log_step("step2","[检查文件路径是否为空]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或文件不存在",self.name)

        # 3.使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)

        # 4. 获取上传文件的后缀
        suffix = path.suffix.lower()

        # 5. 判断文件的后缀
        if suffix =='.pdf':
            state['is_pdf_read_enabled']=True
            state['pdf_path']=import_file_path
        elif suffix =='.md':
            state['is_md_read_enabled']=True
            state['md_path']=import_file_path
        else :
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"不支持的文件类型:{suffix}",self.name)

        # 6. 获取文件的标题名
        file_title = path.stem
        state['file_title']=file_title

        # 7. 返回state
        return state

if __name__ =="__main__":
    setup_logging()

    # 方式：直接实例该节点对象，调用 process 方法
    # 1. 构建该节点需要的 state
    input_state = {
        "file_dir": r"E:\11\temp_dir",
        "import_file_path": r"E:\11\doc\hak180产品安全手册.pdf",
    }

    # 2. 实例 EntryNode 节点
    entry = EntryNode()

    # node_state = entry.process(input_state)  # 不父类执行__call__()
    # node_state =  entry.__call__(input_state)  # 调用父类__call__()
    node_state = entry(input_state)  # 自动调用父类__call__()

    # 序列化打印
    print(json.dumps(node_state,indent=4,ensure_ascii=False))