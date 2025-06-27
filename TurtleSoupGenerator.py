import os
import json
import asyncio
import tkinter as tk
from tkinter import scrolledtext, font
from typing import Dict
from jsonschema import validate, ValidationError
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType, RoleType
from camel.models import ModelFactory
import re
from threading import Thread

from dotenv import load_dotenv

# 加载 api_key.env 文件
load_dotenv(dotenv_path="api_key.env")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("未找到 SiliconFlow API 密钥，请在 api_key.env 文件中配置 SILICONFLOW_API_KEY")


# 海龟汤 JSON Schema 定义
TURTLE_SOUP_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "minLength": 5, "maxLength": 50},
        "description": {"type": "string", "minLength": 30, "maxLength": 1000},
        "answer": {"type": "string", "minLength": 20, "maxLength": 1000},
        "hints": {
            "type": "array",
            "items": {"type": "string", "minLength": 10, "maxLength": 200},
            "minItems": 1,
            "maxItems": 5
        }
    },
    "required": ["title", "description", "answer", "hints"]
}

# 结果比较 JSON Schema
TURTLE_SOUP_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 100.0},
        "compare": {"type": "string", "minLength": 50, "maxLength": 1000},
        "keywords": {
            "type": "array",
            "items": {"type": "string", "minLength": 2, "maxLength": 100},
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True
        }
    },
    "required": ["score", "compare", "keywords"],
    "additionalProperties": False
}

TURTLE_SOUP_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "enum": ["是", "否", "无关", "这个问题无法使用是、否或无关回答，请重新提问"]
        }
    },
    "required": ["response"],
    "additionalProperties": False
}

class TurtleSoupGenerator:
    def __init__(self, output_text: scrolledtext.ScrolledText):
        """初始化 CAMEL-AI 聊天代理，使用 SiliconFlow DeepSeek V3 模型"""
        self.output_text = output_text
        self.generate_model = ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=ModelType.SILICONFLOW_DEEPSEEK_V3,
            model_config_dict={"temperature": 1.2, "top_p": 0.9}
        )
        self.generate_agent = ChatAgent(
            system_message=BaseMessage(
                role_name="TurtleSoupDesigner",
                role_type=RoleType.ASSISTANT,
                meta_dict={},
                content=(
                    "You are a creative designer specializing in generating Turtle Soup puzzles. "
                    "The puzzles should be in Chinese. "
                    "Each puzzle should have a concise title, a mysterious description, a clear answer, "
                    "and 1-5 hints to guide players. Output must be in JSON format."
                )
            ),
            model=self.generate_model
        )

        self.check_model = ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=ModelType.SILICONFLOW_DEEPSEEK_V3,
            model_config_dict={"temperature": 0.5, "top_p": 0.7}
        )
        self.check_agent = ChatAgent(
            system_message=BaseMessage(
                role_name="AnswerEvaluator",
                role_type=RoleType.ASSISTANT,
                meta_dict={},
                content=(
                    "You are an expert evaluator for Turtle Soup puzzles. "
                    "Given a puzzle and a user answer, evaluate the answer by providing: "
                    "1. A score (0.0-100.0) indicating correctness. "
                    "2. A compare field (50-500 words in Chinese) explaining the evaluation. "
                    "3. A list of 1-5 keywords (2-10 words each in Chinese) summarizing key points of the puzzle answer. "
                    "Output must be in Chinese. "
                    "Output must be in JSON format."
                )
            ),
            model=self.check_model
        )

        self.question_model = ModelFactory.create(
            model_platform=ModelPlatformType.SILICONFLOW,
            model_type=ModelType.SILICONFLOW_DEEPSEEK_V3,
            model_config_dict={"temperature": 0.3, "top_p": 0.5}
        )
        self.question_agent = ChatAgent(
            system_message=BaseMessage(
                role_name="QuestionResponder",
                role_type=RoleType.ASSISTANT,
                meta_dict={},
                content=(
                    "You are a Turtle Soup game host. Given a puzzle's answer and a player's question, "
                    "determine if the question is relevant to the answer and respond with '是', '否', or '无关'. "
                    "If the question couldn't be responded with '是', '否', or '无关', respond the question with '这个问题无法使用是、否或无关回答，请重新提问'. "
                    "Output must be in JSON format: {\"response\": \"是\"} or {\"response\": \"否\"} or {\"response\": \"无关\"} or {\"response\": \"这个问题无法使用是、否或无关回答，请重新提问\"}."
                )
            ),
            model=self.question_model
        )

        self.max_retries = 5
        self.current_puzzle = None
        self.used_hints = []

    def preprocess_input(self, user_input: str) -> str:
        """输入预处理：过滤敏感词和指令注入"""
        sensitive_words = ["暴力", "色情", "政治", "低俗"]
        for word in sensitive_words:
            if word in user_input.lower():
                raise ValueError(f"输入包含敏感词：{word}")
        if re.search(r"system:.*|ignore.*previous", user_input, re.IGNORECASE):
            raise ValueError("检测到可能的指令注入")
        return user_input.strip()

    async def stream_text(self, text: str):
        """流式输出文本到 GUI 的输出区域"""
        self.output_text.config(state='normal')
        for char in str(text):
            self.output_text.insert(tk.END, char)
            self.output_text.see(tk.END)
            self.output_text.update()
            await asyncio.sleep(0.02)
        self.output_text.insert(tk.END, "\n")
        self.output_text.config(state='disabled')

    async def generate_soup(self, user_prompt: str, temperature: float = 1.2, top_p: float = 0.9) -> Dict:
        """生成海龟汤并返回结构化输出"""
        try:
            prompt = self.preprocess_input(user_prompt)
            self.generate_model.model_config_dict.update({"temperature": temperature, "top_p": top_p})
            message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=(
                    f"Generate a Turtle Soup puzzle based on the theme: '{prompt}'. "
                    "Return a JSON object with fields: title, description, answer, hints (array of 1-5 strings). "
                    "The puzzle should be in Chinese."
                )
            )
            
            for attempt in range(self.max_retries):
                try:
                    response = self.generate_agent.step(message)
                    content = response.msgs[0].content
                    content = re.sub(r'^```json\n|\n```$', '', content).strip()
                    output = json.loads(content)
                    validate(instance=output, schema=TURTLE_SOUP_SCHEMA)
                    self.current_puzzle = output
                    self.used_hints = []
                    return output
                except (json.JSONDecodeError, ValidationError) as e:
                    if attempt == self.max_retries - 1:
                        raise Exception("无法生成符合格式的输出")
        except Exception as e:
            raise

    async def handle_question(self, question: str) -> Dict:
        """处理玩家提问，返回是/否/无关"""
        if not self.current_puzzle:
            await self.stream_text("错误：请先生成一个谜题")
            return
        try:
            prompt = (
                f"Puzzle answer: {self.current_puzzle['answer']}\n"
                f"Player question: {question}\n"
                "Determine if the question is relevant to the answer and respond with '是', '否', or '无关'. "
                "If the question couldn't be responded with '是', '否', or '无关', respond the question with '这个问题无法使用是、否或无关回答，请重新提问'. "
                "Return a JSON object: {\"response\": \"是\"} or {\"response\": \"否\"} or {\"response\": \"无关\"} or {\"response\": \"这个问题无法使用是、否或无关回答，请重新提问\"}."
            )
            message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=prompt
            )
            
            for attempt in range(self.max_retries):
                try:
                    response = self.question_agent.step(message)
                    if not response.msgs or not isinstance(response.msgs, list) or len(response.msgs) == 0:
                        raise AttributeError("Question response.msgs is empty or not a list")
                    content = response.msgs[0].content
                    content = re.sub(r'^```json\n|\n```$', '', content).strip()
                    output = json.loads(content)
                    validate(instance=output, schema=TURTLE_SOUP_QUESTION_SCHEMA)
                    return output
                except (json.JSONDecodeError, ValidationError, AttributeError) as e:
                    if attempt == self.max_retries - 1:
                        raise Exception("无法生成符合格式的提问回答")
        except Exception as e:
            raise

    async def evaluate_answer(self, user_answer: str) -> Dict:
        """评估玩家答案并返回结构化输出"""
        if not self.current_puzzle:
            await self.stream_text("错误：请先生成一个谜题")
            return
        try:
            prompt = (
                f"Puzzle answer: {self.current_puzzle['answer']}\n"
                f"User answer: {user_answer}\n"
                "Evaluate the user's answer by providing: "
                "1. A score (0.0-1.0) indicating correctness. "
                "2. A compare field (50-500 words in Chinese) explaining the evaluation. "
                "3. A list of 1-5 keywords (2-10 words each in Chinese) summarizing key points. "
                "Output must be in Chinese and in JSON format."
            )
            message = BaseMessage(
                role_name="User",
                role_type=RoleType.USER,
                meta_dict={},
                content=prompt
            )
            self.stream_output("海龟汤答案：")
            self.stream_output(self.current_puzzle['answer'])
            for attempt in range(self.max_retries):
                try:
                    response = self.check_agent.step(message)
                    content = response.msgs[0].content
                    content = re.sub(r'^```json\n|\n```$', '', content).strip()
                    output = json.loads(content)
                    validate(instance=output, schema=TURTLE_SOUP_ANSWER_SCHEMA)
                    return output
                except (json.JSONDecodeError, ValidationError) as e:
                    if attempt == self.max_retries - 1:
                        raise Exception("无法生成符合格式的答案评估")
        except Exception as e:
            raise

    async def stream_puzzle(self):
        """流式输出谜题的 title 和 description"""
        if not self.current_puzzle:
            await self.stream_text("错误：请先生成一个谜题")
            return
        await self.stream_text("=== 海龟汤谜题 ===")
        for key in ["title", "description"]:
            if key in self.current_puzzle:
                await self.stream_text(f"{key.capitalize()}: {self.current_puzzle[key]}")

    async def stream_output(self, output):
        """流式输出字典或字符串"""
        if isinstance(output, dict):
            for key, value in output.items():
                if isinstance(value, list):
                    await self.stream_text(f"{key.capitalize()}:")
                    for item in value:
                        await self.stream_text(f"- {item}")
                else:
                    await self.stream_text(f"{key.capitalize()}: {value}")
        else:
            await self.stream_text(str(output))

class TurtleSoupGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("海龟汤生成器")
        self.root.geometry("1000x800")
        custom_font = font.Font(family="Microsoft YaHei", size=14, weight="normal")  # 字体大小设置为 14

        # 配置输出文本区域，调整字体、字间距和行间距
        self.output_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            state='disabled',
            height=20,
            font=custom_font,  # 使用自定义字体
            spacing1=5,  # 行前间距（像素）
            spacing2=2,  # 字符与行之间的额外间距（像素）
            spacing3=5   # 行后间距（像素）
        )
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.generator = TurtleSoupGenerator(self.output_text)

        # 输入框架
        input_frame = tk.Frame(self.root)
        input_frame.pack(padx=10, pady=5, fill=tk.X)

        # 主题输入
        tk.Label(input_frame, text="主题：").pack(side=tk.LEFT)
        self.theme_entry = tk.Entry(input_frame)
        self.theme_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(input_frame, text="生成谜题", command=self.generate_puzzle).pack(side=tk.LEFT, padx=5)

        # 提问输入
        tk.Label(input_frame, text="提问：").pack(side=tk.LEFT)
        self.question_entry = tk.Entry(input_frame)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(input_frame, text="提交问题", command=self.ask_question).pack(side=tk.LEFT, padx=5)

        # 答案输入
        tk.Label(input_frame, text="答案：").pack(side=tk.LEFT)
        self.answer_entry = tk.Entry(input_frame)
        self.answer_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(input_frame, text="提交答案", command=self.submit_answer).pack(side=tk.LEFT, padx=5)

        # 提示和清空按钮
        button_frame = tk.Frame(self.root)
        button_frame.pack(padx=10, pady=5, fill=tk.X)
        tk.Button(button_frame, text="获取提示", command=self.get_hint).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="清空", command=self.clear).pack(side=tk.LEFT, padx=5)

        # 异步事件循环
        self.loop = asyncio.get_event_loop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.stream_initial_message()

    def stream_initial_message(self):
        """启动时流式输出欢迎消息"""
        Thread(target=lambda: asyncio.run(self.generator.stream_text(
            "欢迎使用海龟汤生成器！请输入主题（例如：悬疑、奇幻）生成谜题\n"
            "然后通过提问获取线索，或输入答案猜测结果"
        ))).start()

    def generate_puzzle(self):
        """生成谜题并显示用户输入的主题"""
        theme = self.theme_entry.get().strip()
        if not theme:
            Thread(target=lambda: asyncio.run(self.generator.stream_text("错误：请输入主题"))).start()
            return
        Thread(target=lambda: asyncio.run(self._generate_puzzle(theme))).start()

    async def _generate_puzzle(self, theme):
        try:
            await self.generator.stream_text(f"用户输入的主题：{theme}")
            await self.generator.generate_soup(theme)
            await self.generator.stream_puzzle()
            self.theme_entry.delete(0, tk.END)  # 清空主题输入框
        except Exception as e:
            await self.generator.stream_text(f"错误：{str(e)}")

    def ask_question(self):
        """处理提问并显示用户输入的问题"""
        question = self.question_entry.get().strip()
        if not question:
            Thread(target=lambda: asyncio.run(self.generator.stream_text("错误：请输入问题"))).start()
            return
        Thread(target=lambda: asyncio.run(self._ask_question(question))).start()

    async def _ask_question(self, question):
        try:
            await self.generator.stream_text(f"用户提问：{question}")
            response = await self.generator.handle_question(question)
            if response:
                await self.generator.stream_output(f"回答：{response['response']}")
            self.question_entry.delete(0, tk.END)  # 清空提问输入框
        except Exception as e:
            await self.generator.stream_text(f"错误：{str(e)}")

    def get_hint(self):
        """获取提示"""
        Thread(target=lambda: asyncio.run(self._get_hint())).start()

    async def _get_hint(self):
        if not self.generator.current_puzzle:
            await self.generator.stream_text("错误：请先生成一个谜题")
            return
        available_hints = [h for h in self.generator.current_puzzle["hints"] if h not in self.generator.used_hints]
        if available_hints:
            hint = available_hints[0]
            self.generator.used_hints.append(hint)
            await self.generator.stream_output(f"提示：{hint}")
        else:
            await self.generator.stream_output("没有更多提示可用！")

    def submit_answer(self):
        """提交答案并显示用户输入的答案"""
        answer = self.answer_entry.get().strip()
        if not answer:
            Thread(target=lambda: asyncio.run(self.generator.stream_text("错误：请输入答案"))).start()
            return
        Thread(target=lambda: asyncio.run(self._submit_answer(answer))).start()

    async def _submit_answer(self, answer):
        try:
            await self.generator.stream_text(f"用户提交的答案：{answer}")
            evaluation = await self.generator.evaluate_answer(answer)
            if evaluation:
                await self.generator.stream_output("=== 答案评估结果 ===")
                await self.generator.stream_output(evaluation)
            self.answer_entry.delete(0, tk.END)  # 清空答案输入框
        except Exception as e:
            await self.generator.stream_text(f"错误：{str(e)}")

    def clear(self):
        """清空输入和输出"""
        self.theme_entry.delete(0, tk.END)
        self.question_entry.delete(0, tk.END)
        self.answer_entry.delete(0, tk.END)
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')
        self.generator.current_puzzle = None
        self.generator.used_hints = []

    def on_closing(self):
        """关闭窗口时停止事件循环"""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()

def main():
    root = tk.Tk()
    app = TurtleSoupGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()