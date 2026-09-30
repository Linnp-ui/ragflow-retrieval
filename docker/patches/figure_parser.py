#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from PIL import Image
# >>> PATCH 2026-08-24 deepdoc-figure-fallback

from common.constants import LLMType
from api.db.services.llm_service import LLMBundle
from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
from common.connection_utils import timeout
from rag.app.picture import vision_llm_chunk as picture_vision_llm_chunk
from rag.prompts.generator import vision_llm_figure_describe_prompt, vision_llm_figure_describe_prompt_with_context
from rag.nlp import append_context2table_image4pdf
# >>> PATCH 2026-08-24 deepdoc-ocr-prompt begin
import re as _ocr_re
# >>> PATCH 2026-08-24 deepdoc-ocr-prompt-v2: 中文OCR提示(实测PaddleOCR-VL最稳)
_OCR_PROMPT = (
    "OCR：提取图片中的所有可见文字与数字，保留单位与布局顺序，只输出提取结果，不要解释。"
)
def _vision_uses_ocr_prompt(vision_model):
    try:
        name = (getattr(vision_model, 'llm_name', '') or '').lower()
        if 'ocr' in name:
            try:
                vision_model.mdl.extra_body = {"temperature": 0}  # >>> PATCH 2026-08-24 deepdoc-ocr-prompt-v2: 稳定OCR输出
            except Exception:
                pass
            return True
    except Exception:
        pass
    return False
def _clean_ocr_text(txt):
    if not txt:
        return txt
    txt = _ocr_re.sub(r'<\|LOC_\d+\|>', '', txt)
    txt = _ocr_re.sub(r'<\|[^|>]{0,40}\|>', '', txt)
    return txt.strip()
# >>> PATCH 2026-08-24 deepdoc-ocr-prompt end
from rag.utils.lazy_image import ensure_pil_image, open_image_for_processing, is_image_like

# need to delete before pr
def vision_figure_parser_figure_data_wrapper(figures_data_without_positions):
    if not figures_data_without_positions:
        return []
    res = []
    for figure_data in figures_data_without_positions:
        img = ensure_pil_image(figure_data[1])
        if not isinstance(img, Image.Image):
            continue
        res.append(
            (
                (img, [figure_data[0]]),
                [(0, 0, 0, 0, 0)],
            )
        )
    return res

def vision_figure_parser_docx_wrapper(sections, tbls, callback=None,**kwargs):
    if not sections:
        return tbls
    try:
        vision_model_config = get_tenant_default_model_by_type(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        vision_model = LLMBundle(kwargs["tenant_id"], vision_model_config)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        figures_data = vision_figure_parser_figure_data_wrapper(sections)
        try:
            docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            boosted_figures = docx_vision_parser(callback=callback)
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    else:
        # >>> PATCH deepdoc-figure-fallback: 无VLM时生成占位，避免图表丢失
        try:
            def is_figure_item(item):
                return is_image_like(item[0][0]) and isinstance(item[0][1], list)
            figures_data = [item for item in tbls if is_figure_item(item)]
            if figures_data:
                fallback=[]
                for it in figures_data:
                    cap=" ".join(it[0][1]) if it[0][1] else ""
                    txt=cap if cap.strip() else "[图表/Figure] 未配置VLM占位"
                    # 保留上下文截断
                    if context_size and sections:
                        try:
                            ctxs=append_context2table_image4pdf(sections,[it],context_size,return_context=True)
                            if ctxs and ctxs[0]:
                                txt=f"{txt} 上下文:{ctxs[0][:200]}"
                        except Exception: pass
                    fallback.append(((it[0][0],[txt]), it[1]))
                tbls=[item for item in tbls if not is_figure_item(item)]
                tbls.extend(fallback)
                callback(0.75, f"[deepdoc-figure-fallback] 无VLM 生成{len(fallback)}个图表占位")
        except Exception as fe:
            logging.warning(f"[deepdoc-figure-fallback] pdf fallback failed: {fe}")
        # <<< END
    return tbls

def vision_figure_parser_figure_xlsx_wrapper(images,callback=None, **kwargs):
    tbls = []
    if not images:
        return []
    try:
        vision_model_config = get_tenant_default_model_by_type(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        vision_model = LLMBundle(kwargs["tenant_id"], vision_model_config)
        callback(0.2, "Visual model detected. Attempting to enhance Excel image extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        figures_data = [((
                        img["image"],   # Image.Image or LazyImage (converted by ensure_pil_image)
                        [img["image_description"]]     # description list (must be list)
                    ),
                    [
                        (0, 0, 0, 0, 0)   # dummy position
                    ]) for img in images]
        try:
            parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
            callback(0.22, "Parsing images...")
            boosted_figures = parser(callback=callback)
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.25, f"Excel visual model error: {e}. Skipping vision enhancement.")
    else:
        # deepdoc-figure-fallback xlsx
        try:
            for im in images:
                desc=im.get('image_description','') or ''
                if not desc or len(desc.strip())<5:
                    im['image_description']=(desc+' [图表占位 无VLM] ').strip()
                    tbls.append(((im['image'], [im['image_description']]), [(0,0,0,0,0)]))
            if images:
                callback(0.23, f"[deepdoc-figure-fallback] xlsx {len(images)} 占位")
        except Exception as fe:
            logging.warning(f"xlsx fallback failed: {fe}")
    return tbls

def vision_figure_parser_pdf_wrapper(tbls, callback=None, **kwargs):
    if not tbls:
        return []
    sections = kwargs.get("sections")
    parser_config = kwargs.get("parser_config", {})
    context_size = max(0, int(parser_config.get("image_context_size", 0) or 0))
    try:
        vision_model_config = get_tenant_default_model_by_type(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        vision_model = LLMBundle(kwargs["tenant_id"], vision_model_config)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:

        def is_figure_item(item):
            return is_image_like(item[0][0]) and isinstance(item[0][1], list)

        figures_data = [item for item in tbls if is_figure_item(item)]
        figure_contexts = []
        if sections and figures_data and context_size > 0:
            figure_contexts = append_context2table_image4pdf(
                sections,
                figures_data,
                context_size,
                return_context=True,
            )
        try:
            docx_vision_parser = VisionFigureParser(
                vision_model=vision_model,
                figures_data=figures_data,
                figure_contexts=figure_contexts,
                context_size=context_size,
                **kwargs,
            )
            boosted_figures = docx_vision_parser(callback=callback)
            tbls = [item for item in tbls if not is_figure_item(item)]
            tbls.extend(boosted_figures)
        except Exception as e:
            callback(0.8, f"Visual model error: {e}. Skipping figure parsing enhancement.")
    return tbls


def vision_figure_parser_docx_wrapper_naive(chunks, idx_lst, callback=None, **kwargs):
    if not chunks:
        return []
    try:
        vision_model_config = get_tenant_default_model_by_type(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
        vision_model = LLMBundle(kwargs["tenant_id"], vision_model_config)
        callback(0.7, "Visual model detected. Attempting to enhance figure extraction...")
    except Exception:
        vision_model = None
    if vision_model:
        @timeout(30, 3)
        def worker(idx, ck):
            img, close_after = open_image_for_processing(ck.get("image"), allow_bytes=True)
            if not isinstance(img, Image.Image):
                return idx, ""
            context_above = ck.get("context_above", "")
            context_below = ck.get("context_below", "")
            if context_above or context_below:
                prompt = vision_llm_figure_describe_prompt_with_context(
                    # context_above + caption if any
                    context_above=ck.get("context_above") + ck.get("text", ""),
                    context_below=ck.get("context_below"),
                )
                logging.info(f"[VisionFigureParser] figure={idx} context_above_len={len(context_above)} context_below_len={len(context_below)} prompt=with_context")
                logging.info(f"[VisionFigureParser] figure={idx} context_above_snippet={context_above[:512]}")
                logging.info(f"[VisionFigureParser] figure={idx} context_below_snippet={context_below[:512]}")
            else:
                prompt = vision_llm_figure_describe_prompt()
                logging.info(f"[VisionFigureParser] figure={idx} context_len=0 prompt=default")

            try:
                description_text = picture_vision_llm_chunk(
                    binary=img,
                    vision_model=vision_model,
                    prompt=prompt,
                    callback=callback,
                )
                return idx, description_text
            finally:
                if close_after and isinstance(img, Image.Image):
                    try:
                        img.close()
                    except Exception:
                        pass

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(worker, idx, chunks[idx])
                for idx in idx_lst
            ]

            for future in as_completed(futures):
                idx, description = future.result()
                description = _clean_ocr_text(description)  # >>> PATCH 2026-08-24 deepdoc-ocr-prompt: 清理版面token
                if description:
                    chunks[idx]['text'] += description
    else:
        # >>> PATCH deepdoc-figure-fallback naive
        try:
            for idx in idx_lst:
                ck=chunks[idx]
                cap=ck.get('text','') or ck.get('context_above','') or ''
                cap=cap.strip()[:80]
                placeholder=f"[图表/Figure] {cap if cap else '未命名图表'} 无VLM占位 已保留可检索 下载原图查看"
                if not ck.get('text') or len(ck.get('text',''))<5:
                    ck['text']=(ck.get('text','')+' '+placeholder).strip()
            callback(0.75, f"[deepdoc-figure-fallback] docx naive 无VLM 生成{len(idx_lst)}个占位")
        except Exception as fe:
            logging.warning(f"[deepdoc-figure-fallback] naive fallback failed: {fe}")
        # <<< END
    
shared_executor = ThreadPoolExecutor(max_workers=10)    

class VisionFigureParser:
    def __init__(self, vision_model, figures_data, *args, **kwargs):
        self.vision_model = vision_model
        self.figure_contexts = kwargs.get("figure_contexts") or []
        self.context_size = max(0, int(kwargs.get("context_size", 0) or 0))
        self._extract_figures_info(figures_data)
        assert len(self.figures) == len(self.descriptions)
        assert not self.positions or (len(self.figures) == len(self.positions))

    def _extract_figures_info(self, figures_data):
        self.figures = []
        self.descriptions = []
        self.positions = []

        for item in figures_data:
            # position
            if len(item) == 2 and isinstance(item[0], tuple) and len(item[0]) == 2 and isinstance(item[1], list) and isinstance(item[1][0], tuple) and len(item[1][0]) == 5:
                img_desc = item[0]
                img = ensure_pil_image(img_desc[0])
                if img is None:
                    continue
                assert len(img_desc) == 2 and isinstance(img_desc[1], list), "Should be (figure, [description])"
                self.figures.append(img)
                self.descriptions.append(img_desc[1])
                self.positions.append(item[1])
            else:
                img = ensure_pil_image(item[0])
                if img is None:
                    continue
                assert len(item) == 2 and isinstance(item[1], list), f"Unexpected form of figure data: get {len(item)=}, {item=}"
                self.figures.append(img)
                self.descriptions.append(item[1])

    def _assemble(self):
        self.assembled = []
        self.has_positions = len(self.positions) != 0
        for i in range(len(self.figures)):
            figure = self.figures[i]
            desc = self.descriptions[i]
            pos = self.positions[i] if self.has_positions else None

            figure_desc = (figure, desc)

            if pos is not None:
                self.assembled.append((figure_desc, pos))
            else:
                self.assembled.append((figure_desc,))

        return self.assembled

    def __call__(self, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        @timeout(30, 3)
        def process(figure_idx, figure_binary):
            context_above = ""
            context_below = ""
            if figure_idx < len(self.figure_contexts):
                context_above, context_below = self.figure_contexts[figure_idx]
            if _vision_uses_ocr_prompt(self.vision_model):
                prompt = _OCR_PROMPT  # >>> PATCH 2026-08-24 deepdoc-ocr-prompt: OCR模型用简明提示
            elif context_above or context_below:
                prompt = vision_llm_figure_describe_prompt_with_context(
                    context_above=context_above,
                    context_below=context_below,
                )
                logging.info(f"[VisionFigureParser] figure={figure_idx} context_size={self.context_size} context_above_len={len(context_above)} context_below_len={len(context_below)} prompt=with_context")
                logging.info(f"[VisionFigureParser] figure={figure_idx} context_above_snippet={context_above[:512]}")
                logging.info(f"[VisionFigureParser] figure={figure_idx} context_below_snippet={context_below[:512]}")
            else:
                prompt = vision_llm_figure_describe_prompt()
                logging.info(f"[VisionFigureParser] figure={figure_idx} context_size={self.context_size} context_len=0 prompt=default")
            if _vision_uses_ocr_prompt(self.vision_model):
                logging.info(f"[VisionFigureParser] figure={figure_idx} model={getattr(self.vision_model,'llm_name','')} -> OCR prompt")
            description_text = picture_vision_llm_chunk(
                binary=figure_binary,
                vision_model=self.vision_model,
                prompt=prompt,
                callback=callback,
            )
            return figure_idx, description_text

        futures = []
        for idx, img_binary in enumerate(self.figures or []):
            futures.append(shared_executor.submit(process, idx, img_binary))

        for future in as_completed(futures):
            figure_num, txt = future.result()
            if txt:
                txt = _clean_ocr_text(txt)  # >>> PATCH 2026-08-24 deepdoc-ocr-prompt: 清理版面token
                if txt:
                    self.descriptions[figure_num] = txt + "\n".join(self.descriptions[figure_num])

        self._assemble()

        return self.assembled
