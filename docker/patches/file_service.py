#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
import asyncio
import base64
import datetime
import io
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

import xxhash
from peewee import fn

from api.db import KNOWLEDGEBASE_FOLDER_NAME, SKILLS_FOLDER_NAME, FileType
from api.db.db_models import DB, Document, File, File2Document, Knowledgebase, Task
from api.db.services import duplicate_name
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from common.misc_utils import get_uuid
from common.ssrf_guard import assert_url_is_safe
from common.constants import TaskStatus, FileSource, ParserType, MAXIMUM_PAGE_NUMBER
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import TaskService
from api.utils.file_utils import filename_type, read_potential_broken_pdf, thumbnail_img, sanitize_path
from rag.llm.cv_model import GptV4
from common import settings



# [XLSX-AUTO-SPLIT-20260825] ─────────────────────────────────────────────────────────────
# 上传多 sheet Excel 自动拆表：每 sheet 一个单 sheet 文档 + table 行级分块
# 清洗规则对齐本地运维脚本 ragflow-eval/split_xlsx_by_sheet.js
# env: RAGFLOW_XLSX_AUTO_SPLIT (默认 1) / RAGFLOW_XLSX_SPLIT_MAX_SHEETS (默认 20)
# ─────────────────────────────────────────────────────────────
def _xlsx_split_enabled() -> bool:
    return os.environ.get('RAGFLOW_XLSX_AUTO_SPLIT', '1').strip().lower() not in ('0', 'false', 'no', 'off')

def _xlsx_split_max_sheets() -> int:
    try:
        return max(2, int(os.environ.get('RAGFLOW_XLSX_SPLIT_MAX_SHEETS', '20')))
    except ValueError:
        return 20

def _xlsx_fmt_cell(v):
    if v is None:
        return ''
    if isinstance(v, datetime.datetime):
        if (v.hour, v.minute, v.second, v.microsecond) == (0, 0, 0, 0):
            return v.date().isoformat()
        return v.isoformat(sep=' ')
    if isinstance(v, datetime.date):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()

def _xlsx_clean_sheet_rows(ws):
    # 1) 去全空行（源表首尾常有样式空行，会产生空 chunk）
    data = []
    for row in ws.iter_rows(values_only=True):
        cells = [_xlsx_fmt_cell(c) for c in row]
        if any(cells):
            data.append(cells)
    if not data:
        return []
    header, body = data[0], data[1:]
    # 2) 无名列：整列无数据 -> 丢弃；有数据 -> 补表头 补充说明N（避免 table 输出 None：）
    max_cols = max(len(r) for r in data)
    keep, renamed, counter = [], {}, 0
    for i in range(max_cols):
        h = header[i] if i < len(header) else ''
        has_data = any(i < len(r) and r[i] for r in body)
        if not h:
            if not has_data:
                continue
            counter += 1
            renamed[i] = '补充说明%d' % counter
        keep.append(i)
    out_header = [renamed.get(i, header[i] if i < len(header) else '') for i in keep]
    out_body = [[(r[i] if i < len(r) else '') for i in keep] for r in body]
    return [out_header] + out_body

def _xlsx_rows_to_bytes(rows, sheet_title) -> bytes:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_title[:31]
    for r in rows:
        ws.append(list(r))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

def split_xlsx_by_sheet(blob, filename):
    """多 sheet xlsx/xlsm -> [(子文档名, 单sheet xlsx bytes)]；不拆时返回 None（关闭/单sheet/超上限/解析失败）。"""
    if not _xlsx_split_enabled():
        return None
    if not re.search(r'\.xlsx$|\.xlsm$', filename, re.IGNORECASE):
        return None
    try:
        from openpyxl import load_workbook
        # [XLSX-SPLIT-FIX-20260826] 不用 read_only：WPS 等工具写出的 <dimension ref="A1"/> 失真，read_only 模式信任 dimension 只出 1 行致空表误判
        wb = load_workbook(io.BytesIO(blob), data_only=True)
    except Exception as e:
        logging.warning('[XLSX-AUTO-SPLIT] %s: not a valid xlsx, skip split: %s', filename, e)
        return None
    try:
        sheet_names = list(wb.sheetnames)
    except Exception as e:
        wb.close()
        logging.warning('[XLSX-AUTO-SPLIT] %s: read sheetnames failed: %s', filename, e)
        return None
    if len(sheet_names) < 2:
        wb.close()
        return None
    if len(sheet_names) > _xlsx_split_max_sheets():
        logging.info('[XLSX-AUTO-SPLIT] %s: %d sheets > max %d, skip split', filename, len(sheet_names), _xlsx_split_max_sheets())
        wb.close()
        return None
    stem = Path(filename).stem
    parts = []
    for idx, sheet_name in enumerate(sheet_names):
        rows = _xlsx_clean_sheet_rows(wb[sheet_name])
        if len(rows) <= 1:
            logging.info('[XLSX-AUTO-SPLIT] %s: sheet %r empty/header-only, skip', filename, sheet_name)
            continue
        safe = re.sub(r'[\\/:*?"<>|\[\]]', '_', sheet_name).strip() or ('sheet%d' % (idx + 1))
        try:
            parts.append(('%s__%s.xlsx' % (stem, safe), _xlsx_rows_to_bytes(rows, sheet_name)))
        except Exception as e:
            logging.warning('[XLSX-AUTO-SPLIT] %s: sheet %r rebuild failed: %s', filename, sheet_name, e)
    wb.close()
    return parts or None


# [AUTO-PARSER-20260826] ─────────────────────────────────────────────────────────────
# 上传时自动选择最佳解析方式（文档级 parser_id），规则见 报告_各格式最佳解析方式.md
# env: RAGFLOW_AUTO_PARSER (默认 1 开启)
# 注: 多 sheet xlsx 走既有自动拆表分支（拆件已 table），不经过此函数
# ─────────────────────────────────────────────────────────────
def _auto_parser_enabled() -> bool:
    return os.environ.get('RAGFLOW_AUTO_PARSER', '1').strip().lower() not in ('0', 'false', 'no', 'off')

def _auto_parser_id(filetype, filename, size, default):
    # docx -> book; pdf>1MB -> manual; pdf<=1MB -> 默认; 单sheet xlsx/xlsm -> table; 其他 -> 默认
    if not _auto_parser_enabled():
        return default
    try:
        name_l = (filename or '').lower()
        if name_l.endswith('.docx'):
            return ParserType.BOOK.value
        if name_l.endswith('.pdf') and size and size > 1000000:
            return ParserType.MANUAL.value
        if name_l.endswith(('.xlsx', '.xlsm')):
            return ParserType.TABLE.value
    except Exception:
        return default
    return default


# [P0-AGG-PREWARM-20260827] 异步预热 agg header/data -> redis + 内存，首问 0.9s -> 0.3s
def _p0_prewarm_enabled():
    return os.environ.get("RAGFLOW_AGG_PREWARM", "1").strip().lower() not in ("0", "false", "no", "off")

def _p0_prewarm_xlsx_async(doc_id, kb_id, location, blob, name):
    if not _p0_prewarm_enabled():
        return
    if not blob or not name.lower().endswith((".xlsx", ".xlsm")):
        return
    def _run():
        try:
            import io as _io
            import openpyxl
            import json as _j
            import os as _o
            # parse header/data (same cleaning as split)
            wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=True)
            ws = wb[wb.sheetnames[-1]]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            wb.close()
            while rows and all((c is None or str(c).strip()=="") for c in rows[0]):
                rows.pop(0)
            if not rows:
                return
            # trimEnd logic
            def _trimEnd(r):
                i=len(r)-1
                while i>=0 and str(r[i]).strip()=="":
                    i-=1
                return r[:i+1] if i>=0 else []
            header=_trimEnd(rows[0])
            data=[]
            for r in rows[1:]:
                t=_trimEnd(r)
                # pad to header len
                while len(t) < len(header):
                    t.append("")
                t=t[:len(header)]
                if all(str(c).strip()=="" for c in t):
                    continue
                data.append(t)
            if not header:
                return
            # redis table store - key matches search._ag_table_execute: agg:table:<doc_id>:<update_date>
            # at upload time update_date ~ create time; use doc_id + "" fallback also
            try:
                try:
                    import valkey as _r
                except ImportError:
                    import redis as _r
                _redis=_r.Redis(host=_o.getenv("REDIS_HOST","redis"), port=int(_o.getenv("REDIS_PORT","6379")), password=_o.getenv("REDIS_PASSWORD") or None, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5)
                _payload=_j.dumps({"header": header, "data": data, "nm": name}, ensure_ascii=False)
                # primary key with doc_id and short date placeholder (search will try exact, then fallback)
                # [HARDEN-P11-STORAGE] 预热写精确键（需 doc 的 update_date，fallback 空键已移除）
                # 尝试取 doc 的 update_date，若无则用空键兼容旧 read（已移除 fallback，故不再写空键）
                try:
                    _dt = ""
                    try:
                        from api.db.services.document_service import DocumentService as _Dq
                        _ok,_d=_Dq.get_by_id(doc_id)
                        if _ok and _d:
                            _dt=str(getattr(_d, "update_date", "") or getattr(_d, "create_date", "") or "")
                    except: pass
                    if _dt:
                        _redis.setex(f"agg:table:{doc_id}:{_dt}", 3600, _payload)
                    else:
                        _redis.setex(f"agg:table:{doc_id}:", 3600, _payload)
                except Exception as _pe:
                    _redis.setex(f"agg:table:{doc_id}:", 3600, _payload)
                # also set with empty date variant for fallback lookup
                print(f"[P0-PREWARM] {name[:24]} rows={len(data)} cols={len(header)}", flush=True)
            except Exception as _e:
                print(f"[P0-PREWARM-REDIS-FAIL] {name[:20]} {_e}", flush=True)
        except Exception as _e:
            print(f"[P0-PREWARM-FAIL] {name[:20]} {type(_e).__name__}: {_e}", flush=True)
    try:
        import threading
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        try:
            _run()
        except Exception:
            pass

class FileService(CommonService):
    # Service class for managing file operations and storage
    model = File

    @classmethod
    @DB.connection_context()
    def get_by_pf_id(cls, tenant_id, pf_id, page_number, items_per_page, orderby, desc, keywords):
        # Get files by parent folder ID with pagination and filtering
        # Args:
        #     tenant_id: ID of the tenant
        #     pf_id: Parent folder ID
        #     page_number: Page number for pagination
        #     items_per_page: Number of items per page
        #     orderby: Field to order by
        #     desc: Boolean indicating descending order
        #     keywords: Search keywords
        # Returns:
        #     Tuple of (file_list, total_count)
        if keywords:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == pf_id), (fn.LOWER(cls.model.name).contains(keywords.lower())), ~(cls.model.id == pf_id))
        else:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == pf_id), ~(cls.model.id == pf_id))
        count = files.count()
        if desc:
            files = files.order_by(cls.model.getter_by(orderby).desc())
        else:
            files = files.order_by(cls.model.getter_by(orderby).asc())

        files = files.paginate(page_number, items_per_page)

        res_files = list(files.dicts())
        for file in res_files:
            if file["type"] == FileType.FOLDER.value:
                file["size"] = cls.get_folder_size(file["id"])
                file["kbs_info"] = []
                children = list(
                    cls.model.select()
                    .where(
                        (cls.model.tenant_id == tenant_id),
                        (cls.model.parent_id == file["id"]),
                        ~(cls.model.id == file["id"]),
                    )
                    .dicts()
                )
                file["has_child_folder"] = any(value["type"] == FileType.FOLDER.value for value in children)
                continue
            kbs_info = cls.get_kb_id_by_file_id(file["id"])
            file["kbs_info"] = kbs_info

        return res_files, count

    @classmethod
    @DB.connection_context()
    def get_kb_id_by_file_id(cls, file_id):
        # Get dataset IDs associated with a file
        # Args:
        #     file_id: File ID
        # Returns:
        #     List of dictionaries containing dataset IDs and names
        kbs = (
            cls.model.select(*[Knowledgebase.id, Knowledgebase.name, File2Document.document_id])
            .join(File2Document, on=(File2Document.file_id == file_id))
            .join(Document, on=(File2Document.document_id == Document.id))
            .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
            .where(cls.model.id == file_id)
        )
        if not kbs:
            return []
        kbs_info_list = []
        for kb in list(kbs.dicts()):
            kbs_info_list.append({"kb_id": kb["id"], "kb_name": kb["name"], "document_id": kb["document_id"]})
        return kbs_info_list

    @classmethod
    @DB.connection_context()
    def get_by_pf_id_name(cls, id, name):
        # Get file by parent folder ID and name
        # Args:
        #     id: Parent folder ID
        #     name: File name
        # Returns:
        #     File object or None if not found
        file = cls.model.select().where((cls.model.parent_id == id) & (cls.model.name == name))
        if file.count():
            e, file = cls.get_by_id(file[0].id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
            return file
        return None

    @classmethod
    @DB.connection_context()
    def get_id_list_by_id(cls, id, name, count, res):
        # Recursively get list of file IDs by traversing folder structure
        # Args:
        #     id: Starting folder ID
        #     name: List of folder names to traverse
        #     count: Current depth in traversal
        #     res: List to store results
        # Returns:
        #     List of file IDs
        if count < len(name):
            file = cls.get_by_pf_id_name(id, name[count])
            if file:
                res.append(file.id)
                return cls.get_id_list_by_id(file.id, name, count + 1, res)
            else:
                return res
        else:
            return res

    @classmethod
    @DB.connection_context()
    def get_all_innermost_file_ids(cls, folder_id, result_ids):
        # Get IDs of all files in the deepest level of folders
        # Args:
        #     folder_id: Starting folder ID
        #     result_ids: List to store results
        # Returns:
        #     List of file IDs
        subfolders = cls.model.select().where(cls.model.parent_id == folder_id)
        if subfolders.exists():
            for subfolder in subfolders:
                cls.get_all_innermost_file_ids(subfolder.id, result_ids)
        else:
            result_ids.append(folder_id)
        return result_ids

    @classmethod
    @DB.connection_context()
    def get_all_file_ids_by_tenant_id(cls, tenant_id):
        fields = [cls.model.id]
        files = cls.model.select(*fields).where(cls.model.tenant_id == tenant_id)
        files.order_by(cls.model.create_time.asc())
        offset, limit = 0, 100
        res = []
        while True:
            file_batch = files.offset(offset).limit(limit)
            _temp = list(file_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def create_folder(cls, file, parent_id, name, count, tenant_id, created_by):
        # Recursively create folder structure
        # Args:
        #     file: Current file object
        #     parent_id: Parent folder ID
        #     name: List of folder names to create
        #     count: Current depth in creation
        #     tenant_id: Tenant ID
        #     created_by: Created by user ID
        # Returns:
        #     Created file object
        if count > len(name) - 2:
            return file
        else:
            file = cls.insert(
                {"id": get_uuid(), "parent_id": parent_id, "tenant_id": tenant_id, "created_by": created_by, "name": name[count], "location": "", "size": 0, "type": FileType.FOLDER.value}
            )
            return cls.create_folder(file, file.id, name, count + 1, tenant_id, created_by)

    @classmethod
    @DB.connection_context()
    def is_parent_folder_exist(cls, parent_id):
        # Check if parent folder exists
        # Args:
        #     parent_id: Parent folder ID
        # Returns:
        #     Boolean indicating if folder exists
        parent_files = cls.model.select().where(cls.model.id == parent_id)
        if parent_files.count():
            return True
        cls.delete_folder_by_pf_id(parent_id)
        return False

    @classmethod
    @DB.connection_context()
    def get_root_folder(cls, tenant_id):
        # Get or create root folder for tenant
        # Args:
        #     tenant_id: Tenant ID
        # Returns:
        #     Root folder dictionary
        for file in cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)):
            return file.to_dict()

        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": file_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": "/",
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def get_kb_folder(cls, tenant_id):
        # Get dataset folder for tenant
        # Args:
        #     tenant_id: Tenant ID
        # Returns:
        #     Knowledge base folder dictionary
        root_folder = cls.get_root_folder(tenant_id)
        root_id = root_folder["id"]
        kb_folder = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == root_id), (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)).first()
        if not kb_folder:
            kb_folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)
            return kb_folder
        return kb_folder.to_dict()

    @classmethod
    @DB.connection_context()
    def new_a_file_from_kb(cls, tenant_id, name, parent_id, ty=FileType.FOLDER.value, size=0, location=""):
        # Create a new file from dataset
        # Args:
        #     tenant_id: Tenant ID
        #     name: File name
        #     parent_id: Parent folder ID
        #     ty: File type
        #     size: File size
        #     location: File location
        # Returns:
        #     Created file dictionary
        for file in cls.query(tenant_id=tenant_id, parent_id=parent_id, name=name):
            return file.to_dict()
        file = {
            "id": get_uuid(),
            "parent_id": parent_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": name,
            "type": ty,
            "size": size,
            "location": location,
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def init_skills_folder(cls, root_id, tenant_id):
        # Initialize skills folder if not exists
        # Args:
        #     root_id: Root folder ID
        #     tenant_id: Tenant ID
        for _ in cls.model.select().where((cls.model.name == SKILLS_FOLDER_NAME) & (cls.model.parent_id == root_id)):
            return
        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": root_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": SKILLS_FOLDER_NAME,
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)

    @classmethod
    @DB.connection_context()
    def init_knowledgebase_docs(cls, root_id, tenant_id):
        # Initialize dataset documents
        # Args:
        #     root_id: Root folder ID
        #     tenant_id: Tenant ID
        for _ in cls.model.select().where((cls.model.name == KNOWLEDGEBASE_FOLDER_NAME) & (cls.model.parent_id == root_id)):
            return
        folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)

        for kb in Knowledgebase.select(*[Knowledgebase.id, Knowledgebase.name]).where(Knowledgebase.tenant_id == tenant_id):
            kb_folder = cls.new_a_file_from_kb(tenant_id, kb.name, folder["id"])
            for doc in DocumentService.query(kb_id=kb.id):
                FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], tenant_id)

    @classmethod
    @DB.connection_context()
    def get_parent_folder(cls, file_id):
        # Get parent folder of a file
        # Args:
        #     file_id: File ID
        # Returns:
        #     Parent folder object
        file = cls.model.select().where(cls.model.id == file_id)
        if file.count():
            e, file = cls.get_by_id(file[0].parent_id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
        else:
            raise RuntimeError("Database error (File doesn't exist)!")
        return file

    @classmethod
    @DB.connection_context()
    def get_all_parent_folders(cls, start_id):
        # Get all parent folders in path
        # Args:
        #     start_id: Starting file ID
        # Returns:
        #     List of parent folder objects
        parent_folders = []
        current_id = start_id
        while current_id:
            e, file = cls.get_by_id(current_id)
            if e and file.parent_id != file.id:
                parent_folders.append(file)
                current_id = file.parent_id
            else:
                parent_folders.append(file)
                break
        return parent_folders

    @classmethod
    @DB.connection_context()
    def insert(cls, file):
        # Insert a new file record
        # Args:
        #     file: File data dictionary
        # Returns:
        #     Created file object
        if not cls.save(**file):
            raise RuntimeError("Database error (File)!")
        return File(**file)

    @classmethod
    @DB.connection_context()
    def delete(cls, file):
        #
        return cls.delete_by_id(file.id)

    @classmethod
    @DB.connection_context()
    def delete_by_pf_id(cls, folder_id):
        return cls.model.delete().where(cls.model.parent_id == folder_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_folder_by_pf_id(cls, user_id, folder_id):
        try:
            files = cls.model.select().where((cls.model.tenant_id == user_id) & (cls.model.parent_id == folder_id))
            for file in files:
                cls.delete_folder_by_pf_id(user_id, file.id)
            return (cls.model.delete().where((cls.model.tenant_id == user_id) & (cls.model.id == folder_id)).execute(),)
        except Exception:
            logging.exception("delete_folder_by_pf_id")
            raise RuntimeError("Database error (File retrieval)!")

    @classmethod
    @DB.connection_context()
    def get_file_count(cls, tenant_id):
        files = cls.model.select(cls.model.id).where(cls.model.tenant_id == tenant_id)
        return len(files)

    @classmethod
    @DB.connection_context()
    def get_folder_size(cls, folder_id):
        size = 0

        def dfs(parent_id):
            nonlocal size
            for f in cls.model.select(*[cls.model.id, cls.model.size, cls.model.type]).where(cls.model.parent_id == parent_id, cls.model.id != parent_id):
                size += f.size
                if f.type == FileType.FOLDER.value:
                    dfs(f.id)

        dfs(folder_id)
        return size

    @classmethod
    @DB.connection_context()
    def add_file_from_kb(cls, doc, kb_folder_id, tenant_id):
        for _ in File2DocumentService.get_by_document_id(doc["id"]):
            return
        file = {
            "id": get_uuid(),
            "parent_id": kb_folder_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": doc["name"],
            "type": doc["type"],
            "size": doc["size"],
            "location": doc["location"],
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        File2DocumentService.save(**{"id": get_uuid(), "file_id": file["id"], "document_id": doc["id"]})

    @classmethod
    @DB.connection_context()
    def move_file(cls, file_ids, folder_id):
        try:
            cls.filter_update((cls.model.id << file_ids,), {"parent_id": folder_id})
        except Exception:
            logging.exception("move_file")
            raise RuntimeError("Database error (File move)!")

    @classmethod
    @DB.connection_context()
    def upload_document(self, kb, file_objs, user_id, src="local", parent_path: str | None = None, parser_config_override: dict | None = None):
        root_folder = self.get_root_folder(user_id)
        pf_id = root_folder["id"]
        self.init_knowledgebase_docs(pf_id, user_id)
        kb_root_folder = self.get_kb_folder(user_id)
        kb_folder = self.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

        safe_parent_path = sanitize_path(parent_path)

        # Merge parser_config_override with KB parser_config if provided
        base_parser_config = kb.parser_config or {}
        if parser_config_override and isinstance(parser_config_override, dict):
            merged_parser_config = {**base_parser_config, **parser_config_override}
        else:
            merged_parser_config = base_parser_config

        err, files = [], []
        for file in file_objs:
            doc_id = file.id if hasattr(file, "id") else get_uuid()
            e, doc = DocumentService.get_by_id(doc_id)
            if e:
                try:
                    if str(doc.kb_id) != str(kb.id):
                        logging.warning(
                            "Existing document id collision detected for %s: belongs to kb_id=%s, incoming kb_id=%s. "
                            "Skipping update to avoid cross-KB overwrite.",
                            doc_id,
                            doc.kb_id,
                            kb.id,
                        )
                        user_msg = "Existing document id collision with another knowledge base; skipping update."
                        err.append(file.filename + ": " + user_msg)
                        continue
                    blob = file.read()
                    # Connector-supplied fingerprint (e.g. xxhash128(S3 ETag))
                    # takes precedence: for connector-sourced docs the bypass
                    # path uses the fingerprint as content_hash, so reverting
                    # to xxhash128(blob) here would defeat it.
                    incoming_fp = getattr(file, "fingerprint", None)
                    new_hash = incoming_fp or xxhash.xxh128(blob).hexdigest()
                    old_hash = doc.content_hash or ""
                    settings.STORAGE_IMPL.put(kb.id, doc.location, blob, kb.tenant_id)
                    doc.size = len(blob)
                    doc.content_hash = new_hash
                    doc = doc.to_dict()
                    DocumentService.update_by_id(doc["id"], doc)
                    if new_hash != old_hash:
                        files.append((doc, blob))
                except Exception as exc:
                    logging.exception(f"Failed to update document {doc_id}: {exc}")
                    err.append(file.filename + ": " + str(exc))
                continue
            try:
                DocumentService.check_doc_health(kb.tenant_id, file.filename)
                filename = duplicate_name(DocumentService.query, name=file.filename, kb_id=kb.id)
                filetype = filename_type(filename)
                if filetype == FileType.OTHER.value:
                    raise RuntimeError("This type of file has not been supported yet!")

                location = filename if not safe_parent_path else f"{safe_parent_path}/{filename}"
                while settings.STORAGE_IMPL.obj_exist(kb.id, location):
                    location += "_"

                blob = file.read()
                if filetype == FileType.PDF.value:
                    blob = read_potential_broken_pdf(blob)

                # [XLSX-AUTO-SPLIT-20260825] 多 sheet Excel 自动拆表：每 sheet 一个单 sheet 文档 + table 行级分块；原多 sheet 文件不入存储
                split_parts = split_xlsx_by_sheet(blob, filename)
                if split_parts:
                    logging.info('[XLSX-AUTO-SPLIT] %s -> %d docs: %s', filename, len(split_parts), [p[0] for p in split_parts])
                    for sub_name, sub_blob in split_parts:
                        sub_name = duplicate_name(DocumentService.query, name=sub_name, kb_id=kb.id)
                        sub_location = sub_name if not safe_parent_path else f"{safe_parent_path}/{sub_name}"
                        while settings.STORAGE_IMPL.obj_exist(kb.id, sub_location):
                            sub_location += "_"
                        # [HARDEN-P11-STORAGE] 补 tenant_id 保持与更新路径一致，防跨租户 404/泄露
                        settings.STORAGE_IMPL.put(kb.id, sub_location, sub_blob, kb.tenant_id)
                        sub_doc = {
                            "id": get_uuid(),
                            "kb_id": kb.id,
                            "parser_id": ParserType.TABLE.value,
                            "pipeline_id": kb.pipeline_id,
                            "parser_config": merged_parser_config,
                            "created_by": user_id,
                            "type": filetype,
                            "name": sub_name,
                            "source_type": src,
                            "suffix": "xlsx",
                            "location": sub_location,
                            "size": len(sub_blob),
                            "thumbnail": "",
                            "content_hash": xxhash.xxh128(sub_blob).hexdigest(),
                        }
                        DocumentService.insert(sub_doc)
                        FileService.add_file_from_kb(sub_doc, kb_folder["id"], kb.tenant_id)
                        files.append((sub_doc, sub_blob))
                        try:
                            _p0_prewarm_xlsx_async(sub_doc["id"], kb.id, sub_location, sub_blob, sub_name)
                        except Exception:
                            pass
                    continue
                # [HARDEN-P11-STORAGE] 补 tenant_id
                settings.STORAGE_IMPL.put(kb.id, location, blob, kb.tenant_id)


                img = thumbnail_img(filename, blob)
                thumbnail_location = ""
                if img is not None:
                    thumbnail_location = f"thumbnail_{doc_id}.png"
                    settings.STORAGE_IMPL.put(kb.id, thumbnail_location, img)

                incoming_fp = getattr(file, "fingerprint", None)
                # [AUTO-PARSER-20260826] 上传时按格式自动选择最佳解析方式（docx->book / pdf>1MB->manual / 单sheet xlsx->table）
                _default_pid = self.get_parser(filetype, filename, kb.parser_id)
                _auto_pid = _auto_parser_id(filetype, filename, len(blob), _default_pid)
                if _auto_pid != _default_pid:
                    logging.info("[AUTO-PARSER-20260826] %s: parser %s -> %s", filename, _default_pid, _auto_pid)
                doc = {
                    "id": doc_id,
                    "kb_id": kb.id,
                    "parser_id": _auto_pid,
                    "pipeline_id": kb.pipeline_id,
                    "parser_config": merged_parser_config,
                    "created_by": user_id,
                    "type": filetype,
                    "name": filename,
                    "source_type": src,
                    "suffix": Path(filename).suffix.lstrip("."),
                    "location": location,
                    "size": len(blob),
                    "thumbnail": thumbnail_location,
                    "content_hash": incoming_fp or xxhash.xxh128(blob).hexdigest(),
                }
                DocumentService.insert(doc)

                FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
                files.append((doc, blob))
                try:
                    _p0_prewarm_xlsx_async(doc["id"], kb.id, location, blob, filename)
                except Exception:
                    pass
            except Exception as e:
                err.append(file.filename + ": " + str(e))

        return err, files

    @classmethod
    @DB.connection_context()
    def list_all_files_by_parent_id(cls, parent_id):
        try:
            files = cls.model.select().where((cls.model.parent_id == parent_id) & (cls.model.id != parent_id))
            return list(files)
        except Exception:
            logging.exception("list_by_parent_id failed")
            raise RuntimeError("Database error (list_by_parent_id)!")

    @staticmethod
    def parse_docs(file_objs, user_id):
        with ThreadPoolExecutor(max_workers=12) as exe:
            threads = []
            for file in file_objs:
                threads.append(exe.submit(FileService.parse, file.filename, file.read(), False))

            res = []
            for th in threads:
                res.append(th.result())

        return "\n\n".join(res)

    @staticmethod
    def parse(filename, blob, img_base64=True, tenant_id=None, layout_recognize=None):
        from rag.app import audio, email, naive, picture, presentation
        from api.apps import current_user

        def dummy(prog=None, msg=""):
            pass

        FACTORY = {ParserType.PRESENTATION.value: presentation, ParserType.PICTURE.value: picture, ParserType.AUDIO.value: audio, ParserType.EMAIL.value: email}
        parser_config = {"chunk_token_num": 16096, "delimiter": "\n!?;。；！？", "layout_recognize": layout_recognize or "Plain Text"}
        kwargs = {"lang": "English", "callback": dummy, "parser_config": parser_config, "from_page": 0, "to_page": MAXIMUM_PAGE_NUMBER, "tenant_id": current_user.id if current_user else tenant_id}
        file_type = filename_type(filename)
        if img_base64 and file_type == FileType.VISUAL.value:
            return GptV4.image2base64(blob)
        cks = FACTORY.get(FileService.get_parser(filename_type(filename), filename, ""), naive).chunk(filename, blob, **kwargs)
        return f"\n -----------------\nFile: {filename}\nContent as following: \n" + "\n".join([ck["content_with_weight"] for ck in cks])

    @staticmethod
    def get_parser(doc_type, filename, default):
        if doc_type == FileType.VISUAL:
            return ParserType.PICTURE.value
        if doc_type == FileType.AURAL:
            return ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION.value
        if re.search(r"\.(msg|eml)$", filename):
            return ParserType.EMAIL.value
        return default

    @staticmethod
    def get_blob(user_id, location):
        bname = f"{user_id}-downloads"
        return settings.STORAGE_IMPL.get(bname, location)

    @staticmethod
    def put_blob(user_id, location, blob):
        bname = f"{user_id}-downloads"
        return settings.STORAGE_IMPL.put(bname, location, blob)

    @classmethod
    @DB.connection_context()
    def delete_docs(cls, doc_ids, tenant_id):
        root_folder = FileService.get_root_folder(tenant_id)
        pf_id = root_folder["id"]
        FileService.init_knowledgebase_docs(pf_id, tenant_id)
        errors = ""
        kb_table_num_map = {}
        for doc_id in doc_ids:
            try:
                e, doc = DocumentService.get_by_id(doc_id)
                if not e:
                    raise Exception("Document not found!")
                tenant_id = DocumentService.get_tenant_id(doc_id)
                if not tenant_id:
                    raise Exception("Tenant not found!")

                b, n = File2DocumentService.get_storage_address(doc_id=doc_id)

                TaskService.filter_delete([Task.doc_id == doc_id])
                if not DocumentService.remove_document(doc, tenant_id):
                    raise Exception("Database error (Document removal)!")

                f2d = File2DocumentService.get_by_document_id(doc_id)
                deleted_file_count = 0
                if f2d:
                    deleted_file_count = FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
                File2DocumentService.delete_by_document_id(doc_id)
                if deleted_file_count > 0:
                    settings.STORAGE_IMPL.rm(b, n)

                doc_parser = doc.parser_id
                if doc_parser == ParserType.TABLE:
                    kb_id = doc.kb_id
                    if kb_id not in kb_table_num_map:
                        counts = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                        kb_table_num_map[kb_id] = counts
                    kb_table_num_map[kb_id] -= 1
                    if kb_table_num_map[kb_id] <= 0:
                        KnowledgebaseService.delete_field_map(kb_id)
            except Exception as e:
                errors += str(e)

        return errors

    _ALLOWED_SCHEMES = {"http", "https"}

    @staticmethod
    def _validate_url_for_crawl(url: str) -> tuple[str, str]:
        """Raise ValueError if the URL is not safe to crawl (SSRF guard).

        Delegates to :func:`common.ssrf_guard.assert_url_is_safe`, which
        validates the scheme, hostname, and every DNS-resolved address, and
        returns ``(hostname, resolved_ip)`` for DNS pinning.

        Only the scheme and host (and port when present) are forwarded to the
        guard so that credentials or query parameters in *url* are never
        written to the log.
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        port_suffix = f":{parsed.port}" if parsed.port else ""
        redacted = f"{parsed.scheme}://{parsed.hostname}{port_suffix}"
        return assert_url_is_safe(redacted, allowed_schemes=FileService._ALLOWED_SCHEMES)

    @staticmethod
    def upload_info(user_id, file, url: str|None=None):
        def structured(filename, filetype, blob, content_type):
            nonlocal user_id
            if filetype == FileType.PDF.value:
                blob = read_potential_broken_pdf(blob)

            location = get_uuid()
            FileService.put_blob(user_id, location, blob)

            return {
                "id": location,
                "name": filename,
                "size": sys.getsizeof(blob),
                "extension": filename.split(".")[-1].lower(),
                "mime_type": content_type,
                "created_by": user_id,
                "created_at": time.time(),
                "preview_url": None
            }

        if url:
            import requests as _requests
            from urllib.parse import urljoin as _urljoin

            _MAX_CRAWL_REDIRECTS = 10

            # Pre-resolve the full redirect chain so that AsyncWebCrawler never
            # follows a server-sent redirect to an unvalidated (potentially
            # internal) host. Each hop is SSRF-checked before being followed;
            # the validated (hostname, ip) pairs are pinned via Chromium's
            # --host-resolver-rules so the browser cannot re-resolve any of them
            # through a fresh DNS query.
            current_url = url
            current_hostname, current_ip = FileService._validate_url_for_crawl(current_url)
            # Accumulate MAP rules for every hostname we encounter in the chain.
            host_pins: dict[str, str] = {current_hostname: current_ip}

            for _ in range(_MAX_CRAWL_REDIRECTS):
                try:
                    _resp = _requests.get(
                        current_url,
                        timeout=10,
                        allow_redirects=False,
                    )
                except _requests.RequestException as _exc:
                    raise ValueError(f"Failed to fetch {current_url!r}: {_exc}") from _exc

                if _resp.status_code not in (301, 302, 303, 307, 308):
                    break

                _location = _resp.headers.get("Location")
                if not _location:
                    break

                _next_url = _urljoin(current_url, _location)
                _next_hostname, _next_ip = FileService._validate_url_for_crawl(_next_url)
                host_pins[_next_hostname] = _next_ip
                current_url = _next_url
            else:
                raise ValueError(
                    f"Exceeded {_MAX_CRAWL_REDIRECTS} redirects fetching {url!r}"
                )

            # Build a single MAP rule string covering every validated hostname
            # in the redirect chain. Chromium uses the pinned IP for each,
            # skipping DNS entirely and eliminating the rebinding window.
            _map_rules = ",".join(f"MAP {h} {ip}" for h, ip in host_pins.items())

            from crawl4ai import (
                AsyncWebCrawler,
                BrowserConfig,
                CrawlerRunConfig,
                DefaultMarkdownGenerator,
                PruningContentFilter,
                CrawlResult
            )
            filename = re.sub(r"\?.*", "", url.split("/")[-1])
            async def adownload():
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=False,
                    extra_args=[f"--host-resolver-rules={_map_rules}"],
                )
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    crawler_config = CrawlerRunConfig(
                        markdown_generator=DefaultMarkdownGenerator(
                            content_filter=PruningContentFilter()
                        ),
                        pdf=True,
                        screenshot=False
                    )
                    # Use the final resolved URL so the browser starts at the
                    # redirect destination rather than re-following the chain.
                    result: CrawlResult = await crawler.arun(
                        url=current_url,
                        config=crawler_config
                    )
                    return result
            page = asyncio.run(adownload())
            if page.pdf:
                if filename.split(".")[-1].lower() != "pdf":
                    filename += ".pdf"
                return structured(filename, "pdf", page.pdf, page.response_headers["content-type"])

            return structured(filename, "html", str(page.markdown).encode("utf-8"), page.response_headers["content-type"])

        DocumentService.check_doc_health(user_id, file.filename)
        return structured(file.filename, filename_type(file.filename), file.read(), file.content_type)

    @staticmethod
    def get_files(files: Union[None, list[dict]], raw: bool = False, layout_recognize: str = None) -> Union[list[str], tuple[list[str], list[dict]]]:
        if not files:
            return  []
        def image_to_base64(file):
            return "data:{};base64,{}".format(file["mime_type"],
                                        base64.b64encode(FileService.get_blob(file["created_by"], file["id"])).decode("utf-8"))
        with ThreadPoolExecutor(max_workers=5) as exe:
            threads = []
            imgs = []
            for file in files:
                if file["mime_type"].find("image") >=0:
                    if raw:
                        imgs.append(FileService.get_blob(file["created_by"], file["id"]))
                    else:
                        threads.append(exe.submit(image_to_base64, file))
                    continue
                threads.append(exe.submit(FileService.parse, file["name"], FileService.get_blob(file["created_by"], file["id"]), True, file["created_by"], layout_recognize))

            if raw:
                return [th.result() for th in threads], imgs
            else:
                return [th.result() for th in threads]
