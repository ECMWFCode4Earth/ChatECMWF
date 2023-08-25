"""
Goal of these tests is to check that the output is formatted in the correct way, such that
the gradio interface can process it without raising exceptions.
"""
import json
import os
import sys

sys.path.append(".")

import uuid
from urllib.parse import urlsplit

from src.agent import agent
from src.config import Logger


def test_kb():
    with open("tests/tool_knowledge_base.txt", "r") as fin:
        for line in fin.read().split("\n")[:-1]:
            agent.memory.clear()
            output = agent(line)
            assert "output" in output.keys()
            assert "intermediate_steps" in output.keys()
            assert len(output["intermediate_steps"]) > 0
            assert "result" in output["output"].keys()


def test_charts():
    with open("tests/tool_chart_question.txt", "r") as fin:
        for line in fin.read().split("\n")[:-1]:
            agent.memory.clear()
            output = agent(line)
            assert "output" in output.keys()
            assert "intermediate_steps" in output.keys()
            assert len(output["intermediate_steps"]) > 0
            scheme, netloc, path, query, fragment = urlsplit(output["output"])


def test_meteogram():
    with open("tests/tool_meteogram_question.txt", "r") as fin:
        for line in fin.read().split("\n")[:-1]:
            agent.memory.clear()
            output = agent(line)
            assert "output" in output.keys()
            assert "intermediate_steps" in output.keys()
            assert len(output["intermediate_steps"]) > 0
            scheme, netloc, path, query, fragment = urlsplit(output["output"])


def test_cdsapi():
    with open("tests/tool_cdsapi_question.txt", "r") as fin:
        for line in fin.read().split("\n")[:-1]:
            agent.memory.clear()
            output = agent(line)
            assert "output" in output.keys()
            assert "intermediate_steps" in output.keys()
            assert len(output["intermediate_steps"]) > 0
            dumped_output = json.loads(output["output"])
            assert "result" in dumped_output.keys()
            assert "success" in dumped_output.keys()
            assert "request" in dumped_output.keys()
            assert dumped_output["success"]
            assert uuid.UUID(dumped_output["result"])
            assert os.path.exists(f"./data/{dumped_output['result']}/download.grib")
            assert os.path.exists(f"./data/{dumped_output['result']}/map.html")
