"""
This module loads the models and the utilities for running it. It is possible to enable the log to stdout of all LLM interactions
by controlling the ```langchain.debug``` parameter. A subclass of ```langchain.memory.ConversationBufferMemory``` has been implemented for
coping with different output keys from the tools. In particular the RetrievalQA tools and the StructuredTools have different output, which need to be parsed with some logic. This module allows the replacement of the LLM simply by replacing the ```model`` object. See the LLama2 branch for a practical example.
"""
import os

import langchain
from langchain.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from .config import Logger, configs
os.environ["REPLICATE_API_TOKEN"] = configs.REPLICATE_API_TOKEN

langchain.deubg = configs.DEBUG


class MyBuffer(ConversationBufferMemory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_context(self, inputs, outputs):
        if "source_documents" in outputs["output"]:
            self.output_key = "result"
            outputs = outputs["output"]
        else:
            self.output_key = "output"
        super().save_context(inputs, outputs)


chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = MyBuffer(
    memory_key="chat_history",
    input_key="input",
    output_key="output",
    return_messages=True,
)

llm = Replicate(
    model=configs.REPLICATE_MODEL,
    input={"temperature": 0.01, "max_length": 500, "top_p": 1}
)
