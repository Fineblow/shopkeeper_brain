import json
import os
import re
from typing import Tuple, Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    """
    文档切分节点
    处理流程：
        1.读md内容, 参数校验, 换行符统计
        2.按标题切割
    """

    name = "document_split_node"
    def process(self, state:ImportGraphState) ->ImportGraphState | dict:
        """文档切分处理流程"""

        # 1.获取输入进行数据校验
        md_content,file_title,max_content_length,min_content_length=self._get_input_validation(state)

        # 2.标题切分
        sections:List[Dict[str,Any]] = self._split_by_title(file_title,md_content)

        # 3.切分合并：  长切短合
        final_sections:List[Dict[str,Any]] = self._split_and_merge(sections,max_content_length,min_content_length)

        # 4.组装切分
        chunks = self._assemble_chunk(final_sections)

        # 5.日志备份
        # 5.1 日志统计
        self._log_summary(md_content,chunks,max_content_length)
        # 5.2 备份
        self._backup_chunks(state, chunks)

        state['chunks'] = chunks

        return state

    def _get_input_validation(self, state:ImportGraphState)->Tuple[str,str,int,int]:
        """对输入数据进行校验"""
        self.log_step("step1","切分文档的参数校验以及获取...")

        md_content = state.get('md_content')
        if not md_content:
            raise StateFieldError(self.name,"md_content",str)
        file_title = state.get('file_title')
        if not file_title:
            raise StateFieldError(self.name,"file_title",str)
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if max_content_length < 0 or min_content_length <0 or min_content_length >= max_content_length:
            raise ValidationError("切分最大值或最小值参数错误",self.name)

        return md_content,file_title,max_content_length,min_content_length

    def _split_by_title(self, file_title, md_content):
        """根据标题切分文档
               md_content: 整个文档内容
               file_title: 文件标题 不带扩展名
               return [
                   {
                       "parent_title": "# HAK 180",
                       "title": "## HAK 180 烫金机",
                       "body": "产品安全手册（简体中文）...",
                       "file_title": "hak180产品安全手册"
                   }
               ]
        """
        self.log_step("step2","根据标题进行切分...")
        sections = []
        in_fence = False # 是否在围栏内
        heading_re = re.compile(r"^\s*(#{1,6})\s+.+")# 匹配标题正则表达式      (#{1,6})  匹配组     取标题级别：match.group(1)     原文mat
        body:List[str] = [] # 收集正文
        content_lines = md_content.split("\n")
        current_title = ""   # 当前标题
        current_level = 0    # 当前标题级别
        hierarchy = [""]*7  # 标题层级     ["","一级","二级","三级","四级","五级","六级"]

        # 封装section  段落 + 元信息
        def _flush():
            if current_title or body:
                parent_title = ""
                for lev in range(current_level-1,0,-1):
                    if hierarchy[lev]:
                        parent_title = hierarchy[lev]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "parent_title":parent_title,
                    "title":current_title if current_title else file_title,
                    "body":"\n".join(body),
                    "file_title":file_title
                }
                )
        for idx,line in enumerate(content_lines):
            if line.startswith("~~~") or line.startswith("```"):
                in_fence = not in_fence

            # 匹配标题
            match = heading_re.match(line) if not in_fence else None

            #是标题
            if match:
                _flush()    # 帮我把当前标题，之前的内容封装成secion ->  List

                level = len(match.group(1))
                current_level = level
                current_title = line
                hierarchy[current_level] = current_title
                # hierarchy = [""] * 7  # 重置，因为不重置，这里存放的都是上个段落对应的标题。 全部置空是错误的。
                for i in range(level+1,7): # 只清理大于当前级别的子级别，因为这些子级别是上个段落遗留的。当前级别父
                    hierarchy[i] = [""]
                body = []# 置空，只装填当前标题的正文行

            else:
                body.append(line.strip())
        # 处理最后一个段落
        _flush()

        return sections

    def _split_and_merge(self, sections, max_content_length, min_content_length):
        self.log_step("step3", "切分及合并...")

        # 1. 切分
        current_sections = []
        for section in sections:
            #  current_sections.append([{},{}])            [[{},{}],[{},{}],[{},{}]]
            #  current_sections.extend([])                 [{},{},{},{}]
            current_sections.extend(self._split_long_section(section,max_content_length))
        # 2.合并
        final_sections =self._merge_short_sections(current_sections,min_content_length)

        return final_sections

    def _split_long_section(self, section, max_content_length):
        """切分长段落"""
        self.log_step("step3", "进行长内容的切分")

        # 1.获取section属性值
        parent_title = section.get("parent_title")
        title = section.get("title")
        body = section.get("body")
        file_title = section.get("file_title")

        # 2.表格处理
        if "<table>" in body:
            self.logger.info("表格处理中...")
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 最大标题长度限制
        MAX_TITLE_LENGTH = 50
        if len(title) >MAX_TITLE_LENGTH:
            self.logger.warning(f"检测文件{file_title}对应的{title}长度过长...")
            title = title[:MAX_TITLE_LENGTH]

        # 3.定义标题前缀
        title_prefix = title + "\n\n"

        # 4.标题+正文  小于 阈值  ， 不用切分直接返回,但是也存储列表返回。
        if len(title_prefix) + len(body) <= max_content_length:
            return [section]

        # 5.获取body可用长度
        body_length = max_content_length - len(title_prefix)

        sub_sections = []
        # 6.需要切分
        if len(body) > body_length:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？"],
                chunk_size=body_length,# 切片最大长度
                chunk_overlap=0, # 不重叠
                keep_separator=False,  # 不保留分隔符
            )
            texts = splitter.split_text(body)
            if texts<=1:
                return [section]

            for index,text in enumerate(texts):
                sub_sections.append({
                    "parent_title": parent_title,
                    "title": title + f"_{index+1}",
                    "body": text,
                    "file_title": file_title,
                    "part":f"{index+1}"
                })
        return sub_sections

    def _merge_short_sections(self, current_sections, min_content_length):
        """合并短章节内容"""
        final_sections = []

        # 当前section
        current_section = current_sections[0]

        # 用于指向同一个父标题的下一个元素section
        for new_section in current_sections[1:]:
            same_parent = current_section["parent_title"]==new_section["parent_title"]
            if same_parent and len(current_section["body"]) < min_content_length:
                current_section["body"] += "\n\n" + new_section["body"]
                current_section["title"] = current_section["parent_title"]
                current_section["part"] = 0
            else:
                final_sections.append(current_section)
                current_section = new_section

        final_sections.append(current_section)

        # 4. 对所有 section 的 part 做处理
        part_counter={}
        for final_section in final_sections:
            if "part" in final_section:
                parent_title = final_section['parent_title']
                part_counter['parent_title'] = part_counter.get('parent_title',0)+1
                new_part = part_counter['parent_title']
                final_section['part'] = new_part
                final_section['title'] = final_section['title'] + f"-{new_part}"

        return final_sections

    def _assemble_chunk(self, final_sections):
        """最终组装结果"""
        self.log_step("step4", "组装最终的切片信息...")
        chunks = []
        for chunk in final_sections:
            # 1. 获取 chunk 的信息
            title = chunk.get('title')
            body = chunk.get('body')
            parent_title = chunk.get('parent_title')
            file_title = chunk.get('file_title')
            content = f"{title}\n\n{body}"

            # 2. 构建最终 chunk 对象
            assemble_chunk=({
                "title": title,
                "parent_title": parent_title,
                "file_title": file_title,
                "content": content
            })

            # 3. 判断 part 是否存在
            if 'part' in chunk:
                part = chunk.get('part')
                assemble_chunk['part'] = part
            chunks.append(assemble_chunk)
        return chunks

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto\hak180产品安全手册_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        "file_title": "hak180产品安全手册",
        "md_content": content,
        "file_dir": r"D:\workspace\shopkeeper_brain\knowledge\output\temp_dir\hak180产品安全手册\auto"
    }
    document_node.process(state)