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


from api.db import TenantPermission
from api.db.db_models import File, Knowledgebase
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.user_service import TenantService
from common.constants import StatusEnum


def check_kb_manage_permission(kb_id: str, user_id: str) -> tuple[bool, str]:
    """管理/编辑知识库文档的权限检查。

    只有知识库创建者本人可以编辑、管理、下载该知识库的文档；其他用户
    （包括租户 Owner/Admin）一律没有管理和编辑权限。
    """
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    if not e or not kb:
        return False, "Knowledge base not found."

    if kb.status != StatusEnum.VALID.value:
        return False, "Knowledge base is not available."

    if kb.created_by == user_id:
        return True, "OK"

    return False, "You do not have permission to manage documents in this knowledge base."


def check_kb_team_permission(kb: dict | Knowledgebase, other: str) -> bool:
    kb = kb.to_dict() if isinstance(kb, Knowledgebase) else kb

    kb_tenant_id = kb["tenant_id"]

    if kb_tenant_id == other:
        return True

    if kb["permission"] != TenantPermission.TEAM:
        return False

    joined_tenants = TenantService.get_joined_tenants_by_user_id(other)
    return any(tenant["tenant_id"] == kb_tenant_id for tenant in joined_tenants)


def check_file_team_permission(file: dict | File, other: str) -> bool:
    file = file.to_dict() if isinstance(file, File) else file

    file_tenant_id = file["tenant_id"]
    if file_tenant_id == other:
        return True

    file_id = file["id"]

    kb_ids = [kb_info["kb_id"] for kb_info in FileService.get_kb_id_by_file_id(file_id)]

    for kb_id in kb_ids:
        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            continue

        # 知识库文件与知识库文档采用同一套管理权限：
        # 仅知识库创建者或所在租户 Owner/Admin 可访问、下载与管理。
        allowed, _ = check_kb_manage_permission(kb_id, other)
        if allowed:
            return True

    return False
