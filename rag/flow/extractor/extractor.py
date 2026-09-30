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
import json
import logging
import random
import asyncio
from copy import deepcopy

import xxhash

from agent.component.llm import LLMParam, LLM
from rag.flow.base import ProcessBase, ProcessParamBase
from rag.prompts.generator import run_toc_from_text


class ExtractorParam(ProcessParamBase, LLMParam):
    def __init__(self):
        super().__init__()
        self.field_name = ""

    def check(self):
        super().check()
        self.check_empty(self.field_name, "Result Destination")


class Extractor(ProcessBase, LLM):
    component_name = "Extractor"

    async def _build_TOC(self, docs):
        self.callback(0.2,message="Start to generate table of content ...")
        docs = sorted(docs, key=lambda d:(
            d.get("page_num_int", 0)[0] if isinstance(d.get("page_num_int", 0), list) else d.get("page_num_int", 0),
            d.get("top_int", 0)[0] if isinstance(d.get("top_int", 0), list) else d.get("top_int", 0)
        ))
        toc = await run_toc_from_text([d["text"] for d in docs], self.chat_mdl)
        logging.info("------------ T O C -------------\n"+json.dumps(toc, ensure_ascii=False, indent='  '))
        ii = 0
        while ii < len(toc):
            try:
                idx = int(toc[ii]["chunk_id"])
                del toc[ii]["chunk_id"]
                toc[ii]["ids"] = [docs[idx]["id"]]
                if ii == len(toc) -1:
                    break
                for jj in range(idx+1, int(toc[ii+1]["chunk_id"])+1):
                    toc[ii]["ids"].append(docs[jj]["id"])
            except Exception as e:
                logging.exception(e)
            ii += 1

        if toc:
            d = deepcopy(docs[-1])
            d["doc_id"] = self._canvas._doc_id
            d["toc"] = json.dumps(toc, ensure_ascii=False)
            d["content_with_weight"] = json.dumps(toc, ensure_ascii=False)
            d["toc_kwd"] = "toc"
            d["available_int"] = 0
            d["page_num_int"] = [100000000]
            d["id"] = xxhash.xxh64((d["content_with_weight"] + str(d["doc_id"])).encode("utf-8", "surrogatepass")).hexdigest()
            return d
        return None

    async def _invoke(self, **kwargs):
        self.set_output("output_format", "chunks")
        self.callback(random.randint(1, 5) / 100.0, "Start to generate.")
        inputs = {}
        for k, v in self.get_input_elements().items():
            inputs[k] = v["value"] if isinstance(v, dict) else v
        # Pipeline execution passes the upstream component output as kwargs
        # (e.g. ``chunks``), while canvas runs declare inputs explicitly.
        # Merge both so extractors always see their upstream chunks; declared
        # inputs take precedence.
        for k, v in kwargs.items():
            if k not in inputs:
                inputs[k] = v
        chunks = []
        chunks_key = ""
        args = dict(inputs)
        list_inputs = {k: v for k, v in inputs.items() if isinstance(v, list)}
        if "chunks" in list_inputs:
            chunks_key = "chunks"
        elif list_inputs:
            chunks_key = next(iter(list_inputs))
        if chunks_key:
            chunks = deepcopy(list_inputs[chunks_key])

        if chunks:
            if self._param.field_name == "toc":
                for ck in chunks:
                    ck["doc_id"] = self._canvas._doc_id
                    ck["id"] = xxhash.xxh64((ck["text"] + str(ck["doc_id"])).encode("utf-8")).hexdigest()
                toc =await self._build_TOC(chunks)
                chunks.append(toc)
                self.set_output("chunks", chunks)
                return

            prog = 0
            done = 0
            total = len(chunks)
            semaphore = asyncio.Semaphore(4)

            async def extract_one(ck):
                nonlocal prog, done
                text = ck["text"] if isinstance(ck.get("text"), str) else ""
                args[chunks_key] = text
                msg, sys_prompt = self._sys_prompt_and_msg([], args)
                msg.insert(0, {"role": "system", "content": sys_prompt})
                # Prompt templates may only contain a literal placeholder
                # (e.g. "[在此处插入文本]") instead of a {var} reference.
                # Inject the chunk text explicitly so the LLM sees the content.
                if text:
                    for m in msg:
                        if m["role"] == "user" and isinstance(m.get("content"), str):
                            content = m["content"]
                            injected = False
                            for ph in ("[在此处插入文本]", "[文本内容]", "[插入文本]"):
                                if ph in content:
                                    content = content.replace(ph, text)
                                    injected = True
                            if not injected and text not in content:
                                content = content.rstrip() + "\n\n" + text
                            m["content"] = content
                async with semaphore:
                    ck[self._param.field_name] = await self._generate_async(msg)
                done += 1
                prog = done / total
                self.callback(prog, f"{done} / {total}")

            await asyncio.gather(*(extract_one(ck) for ck in chunks))
            self.set_output("chunks", chunks)
        else:
            msg, sys_prompt = self._sys_prompt_and_msg([], args)
            msg.insert(0, {"role": "system", "content": sys_prompt})
            self.set_output("chunks", [{self._param.field_name: await self._generate_async(msg)}])
