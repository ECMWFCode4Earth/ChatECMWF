"""
This module implements the callbacks executed on events triggered by the gradio interface.
"""
import json
import os
import sys
import traceback
from datetime import datetime

import gradio as gr

from ..agent import agent
from ..config import Logger
from ..utils import sources_markdown


def answer_question_using_tools(question):
    """
    This is the main callback, implementing the logic for calling the LLM and formatting the output
    to be employed in the gradio.Chatbot and other objects in the interface.

    Note
    ----
    It should be possible to exploit gradio built-in async engine, having the answer streamed by the
    LLM. For this, the asynchronous implementation of the agent in langchain should be considered.

    Args:
    -----
       question: str
                 the user query

    Returns:
    --------
       answer: str
               the answer passed to the interface
    """
    answer = ""

    try:
        result = agent(question)
        intermediate_steps = result["intermediate_steps"]
        Logger.info(f"ChatGPT action: {result}")
        if len(intermediate_steps) == 0:
            answer += result["output"]
        elif intermediate_steps[0][0].tool == "meteogram":
            url = result["output"]
            answer += "Please find the meteogram below, you can click on it to enlarge it and download it:"
            answer += f'<a href="{url}"><img src="{url}"/></a>'
            answer += "Sources used: ECMWF Meteogram API"
        elif "chart" in intermediate_steps[0][0].tool:
            url = result["output"]
            answer += "Please find the chart below:"
            answer += f'<a href="{url}"><img src="{url}"/></a>'
            answer += "Sources used: ECMWF Charts API"
        elif "cdsapi" in intermediate_steps[0][0].tool:
            output = json.loads(result["output"])
            map_uuid = output["result"]
            answer += f'Look at this map I just generated for you: <a href="/maps/{map_uuid}/map.html">link</a>!\n'
            answer += f'<a href="/maps/{map_uuid}/download.grib">Here</a> you can download the GRIB file.\n'
            iframe = f'<div id="map-frame"> <iframe src="/maps/{map_uuid}/map.html"width="100%" height="280px" title="Map Output"> </iframe> </div>'
            answer += iframe
            if output["success"]:
                answer += "The request I employed for the data: \n"
                answer += '<pre> \n <code class="language-python"> <button class="copy_code_button" title="copy"> \
                <span class="copy-text"><svg viewBox="0 0 32 32" height="100%" width="100%" xmlns="http://www.w3.org/2000/svg"><path d="M28 10v18H10V10h18m0-2H10a2 2 0 0 0-2 2v18a2 2 0 0 0 2 2h18a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2Z" fill="currentColor"></path><path d="M4 18H2V4a2 2 0 0 1 2-2h14v2H4Z" fill="currentColor"></path></svg></span> \
                <span class="check"><svg stroke-linejoin="round" stroke-linecap="round" stroke-width="3" stroke="currentColor" fill="none" viewBox="0 0 24 24" height="100%" width="100%" xmlns="http://www.w3.org/2000/svg"><polyline points="20 6 9 17 4 12"></polyline></svg></span> \
                </button>\n'
                answer += (
                    "{"
                    + "\n".join(
                        "{!r}: {!r},".format(k, v) for k, v in output["request"].items()
                    )
                    + "} \n </code> \n </pre>"
                )
        else:
            answer += result["output"]["result"]
            answer += "\n\nFor this answer, I used the following sources:\n"

            source_field = "link"
            markdown = ""
            try:
                markdown = sources_markdown(result["output"], source_field)
            except:
                source_field = "source"
                markdown = sources_markdown(result["output"], source_field)

            answer += markdown
    except Exception as e:
        Logger.opt(exception=True).info("Exception:")
        answer += f"I was not able to generate an answer due to an internal problem: {sys.exc_info()}"
    return answer


def predict(
    question,
    chatbot,
    history,
):
    """
    Main callback, attaching the answer to the history.
    """
    answer = answer_question_using_tools(question)

    new_interaction = [[question, answer]]
    return history + new_interaction, history + new_interaction, "Status: OK"


def upvote(
    chatbot,
    history,
):
    """
    Feedback callback persisting the upvotes into a local pickle file, named after the UTC datetime
    """
    formatted_dtime = datetime.now().strftime("%Y-%m-%dT%H:%M")
    with open(f"feedbacks/positive_{formatted_dtime}", "w") as fout:
        fout.write(history)
    return (
        history + [["👍", "Thanks for upvoting my answer!"]],
        history + [["👍", "Thanks for upvoting my answer!"]],
        "Status: OK",
    )


def downvote(chatbot, history):
    """
    Feedback callback persisting the downvotes into a local pickle file, named after the UTC datetime
    """
    formatted_dtime = datetime.now().strftime("%Y-%m-%dT%H:%M")
    with open(f"feedbacks/negative_{formatted_dtime}", "w") as fout:
        fout.write(history)
    return (
        history
        + [
            [
                "👎",
                "I am sorry you didn't find my answer useful... Thanks for letting me know!",
            ]
        ],
        history
        + [
            [
                "👎",
                "I am sorry you didn't find my answer useful... Thanks for letting me know!",
            ]
        ],
        "Status: OK",
    )


def reset_state():
    """
    Resetting the interface.
    """
    return [], [], "Reset Done"


def reset_textbox():
    """
    Resetting the textbox.
    """
    return gr.update(value=""), ""


def cancel_outputing():
    """
    Stopping the interface.
    """
    return "Stop Done"


def transfer_input(inputs):
    """
    Updating the value after hitting the button.
    """
    textbox = reset_textbox()
    return (
        inputs,
        gr.update(value=""),
        gr.Button.update(visible=True),
    )
