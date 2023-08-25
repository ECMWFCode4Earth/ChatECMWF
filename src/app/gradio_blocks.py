"""
This module generates the dashboard interface, making use of gradio Blocks API. 
"""
import gradio as gr

from .gradio_functions import (
    cancel_outputing,
    downvote,
    predict,
    reset_state,
    reset_textbox,
    transfer_input,
    upvote,
)
from .theme import description, description_top, small_and_beautiful_theme, title


def build_interface():
    with open("./assets/custom.css", "r", encoding="utf-8") as f:
        customCSS = f.read()

    with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
        history = gr.State([])
        user_question = gr.State("")

        with gr.Row():
            gr.HTML(title)
            status_display = gr.Markdown("Success", elem_id="status_display")

        gr.Markdown(description_top)

        with gr.Row(scale=1).style(equal_height=True):
            with gr.Column(scale=5):
                with gr.Row(scale=1):
                    chatbot = gr.Chatbot(elem_id="chuanhu_chatbot").style(height="100%")

                with gr.Row(scale=1):
                    with gr.Column(scale=12):
                        user_input = gr.Textbox(
                            show_label=False, placeholder="Enter text"
                        ).style(container=False)
                    with gr.Column(min_width=70, scale=1):
                        submitBtn = gr.Button("Send")
                    with gr.Column(min_width=70, scale=1):
                        cancelBtn = gr.Button("Stop")

                with gr.Row(scale=1):
                    emptyBtn = gr.Button("🧹 New conversation")

                    with gr.Row(scale=1).style(equal_height=True):
                        with gr.Column(min_width=70, scale=1):
                            upvoteBtn = gr.Button("👍")
                        with gr.Column(min_width=70, scale=1):
                            downvoteBtn = gr.Button("👎")

            with gr.Column():
                # Here we could have sliders for extra parameters, or maybe an info box.
                pass

        gr.Markdown(description)

        predict_args = dict(
            fn=predict,
            inputs=[
                user_question,
                chatbot,
                history,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )

        reset_args = dict(
            fn=reset_textbox, inputs=[], outputs=[user_input, status_display]
        )

        # Chatbot
        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[user_input],
            outputs=[user_question, user_input, submitBtn],
            show_progress=True,
        )

        predict_event1 = user_input.submit(**transfer_input_args).then(**predict_args)
        predict_event2 = submitBtn.click(**transfer_input_args).then(**predict_args)

        upvote_args = dict(
            fn=upvote,
            inputs=[
                chatbot,
                history,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=False,
        )

        downvote_args = dict(
            fn=downvote,
            inputs=[
                chatbot,
                history,
            ],
            outputs=[chatbot, history, status_display],
            show_progress=False,
        )

        upvote_event = upvoteBtn.click(**upvote_args)
        downvote_event = downvoteBtn.click(**downvote_args)

        emptyBtn.click(
            reset_state,
            outputs=[chatbot, history, status_display],
            show_progress=True,
        )

        emptyBtn.click(**reset_args)

        cancelBtn.click(
            cancel_outputing,
            [],
            [status_display],
            cancels=[predict_event1, predict_event2],
        )
        return demo
