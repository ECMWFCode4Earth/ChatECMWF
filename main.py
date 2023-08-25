"""
This is ChatECMWF, a friendly bot based on LLM, either hosted like OpenAI ChatGPT, or self-hosted as Llama2.
Multiple tools are loaded and available for usage. The main module builds the interface and launch the Gradio app. 
"""
from src.app.gradio_blocks import build_interface
from src.config import Logger, configs

if __name__ == "__main__":
    Logger.info("Initialising gradio interface...")
    demo = build_interface()
    demo.title = "ChatECMWF"
    Logger.info(f"Launching interface on {configs.BIND_IP}:{configs.PORT}...")
    demo.queue(concurrency_count=1).launch(
        server_name=configs.BIND_IP, server_port=configs.PORT
    )
