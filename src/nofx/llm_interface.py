from typing import Tuple, List, Optional

import re
import json
import logging
from openai import OpenAI

from .logger import BaseClassWithLogger
from .structs import Action
from .utils import retry


class LLMInterface(BaseClassWithLogger):
    def __init__(self, api_key: str, model: str, temperature: float, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self.model = model
        self.temperature = temperature

        if model in ["deepseek-chat", "deepseek-reasoner"]:
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            raise NotImplementedError(f"未知模型: {model}")

    @retry(max_retries=3, delay=5.0, output="")
    def call_client(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
        )
        return response.choices[0].message.content

    def parse_ai_response(self, text: str) -> Tuple[str, List[Action]]:
        # 提取 <reasoning> ... </reasoning>
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # 提取 decision 中的 JSON 代码块
        decisions = []
        decision_match = re.search(r"<decision>.*?```json(.*?)```.*?</decision>", text, re.DOTALL)
        if decision_match:
            decision_str = decision_match.group(1).strip()
            try:
                decisions = json.loads(decision_str)
            except json.JSONDecodeError as e:
                self.logger.exception(f"⚠️ 解析JSON文件失败: {e}")
                self.logger.warning(f"原始JSON字符串为:\n{decision_str:s}")

        actions = []
        for decision in decisions:
            action = Action.from_dict(decision)

            if not action:
                self.logger.warning(f"⚠️ 无效的JSON格式:\n{decision}")
                continue

            actions.append(action)

        return reasoning, actions

    def __call__(self, system_prompt: str, user_prompt: str) -> Tuple[str, List[Action]]:
        ai_response = self.call_client(system_prompt, user_prompt)
        reasoning, actions = self.parse_ai_response(ai_response)
        return reasoning, actions
