from datetime import datetime
from time import sleep

from langchain.agents import Tool
from langchain.tools import StructuredTool

from ..dbs import ecmwf_kb, ecmwf_web
from .cdsapi import cdsapi_tool
from .charts import charts
from .meteogram import meteogram

retriever_tools = [
    Tool(
        name="ECWMF web content",
        func=ecmwf_web,
        description="Useful for when you need to answer general questions about ECMWF and its products. The input should be a fully formed question.",
        return_direct=True,
    ),
    Tool(
        name="ECMWF Knowledge base, code base.",
        func=ecmwf_kb,
        description="Useful for when you need to answer questions on software, tutorials, available modules and APIs.",
        return_direct=True,
    ),
]

now = datetime.now()
formatted_now = f"{now.year}-{now.month:02d}-{now.day:02d}T{now.hour:02d}:{now.minute:02d}:{now.second:02d}Z"
meteogram_tool = [
    StructuredTool.from_function(meteogram, name="meteogram", return_direct=True)
]
chart_tool = [
    StructuredTool.from_function(
        charts,
        name="chart",
        description=f"Use this tool when asked to download a chart for a particular product. Time must be passed in YYYY-mm-ddTHH:MM:SSZ format. Keep in mind that todays date and time are {formatted_now}.",
        return_direct=True,
    )
]

cdsapi_tool = [
    StructuredTool.from_function(
        cdsapi_tool,
        name="cdsapi",
        description=f"This tool is employed when asked for data from the climate data services. You must infer which keyword arguments need to be passed to the tool, depending on the suer input. If you do not know which parameters are necessary, ask for more information.  Year, month, day and hours need to be provided as list of integers. Even if the user ask for a single element, always pass the argument wrapped in a list. If a parameter is optional, it means you can call the tool without passing it explicitely. \n\n\n Example: please download me the data of the arctic regional reanalysis for the last 3 years in the January month. \n Answer: I will call the tool with following keyword arguments. product=arctic regional reanalysis, month=[1], year=[2022, 2021, 2020]. \n Bear in mind that todays date and time are {formatted_now}",
        return_direct=True,
    )
]
