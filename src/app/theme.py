# -*- coding:utf-8 -*-
import gradio as gr

title = """<h1 align="left" style="min-width:400px; margin-top:0;"> 🚀 ChatECMWF</h1>"""

description_top = """\
<div align="left">
<p>
This is an expertimental version of ChatECMWF. It can currently retrieve meteograms, charts, reanalysis data from the CDS as well as information from ECMWF's Confluence knowledge base.
</p >
</div>
"""
description = """\
<div align="center" style="margin:16px 0">
&copy; 2023 Piero Ferrarese, Giacomo Bighin as participants in ECMWF’s Code for Earth 2023.
</div>
"""

# accent_color = "255, 193, 96"
accent_color = "100, 149, 237"  # Cornflower blue

small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="rgba(" + accent_color + ", 1.0)",
        c100="rgba(" + accent_color + ", 0.2)",
        c200="rgba(" + accent_color + ", 1.0)",
        c300="rgba(" + accent_color + ", 0.32)",
        c400="rgba(" + accent_color + ", 0.32)",
        c500="rgba(" + accent_color + ", 1.0)",
        c600="rgba(" + accent_color + ", 1.0)",
        c700="rgba(" + accent_color + ", 0.32)",
        c800="rgba(" + accent_color + ", 0.32)",
        c900="rgba(" + accent_color + ", 1.0)",
        c950="rgba(" + accent_color + ", 1.0)",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f9fafb",
        c100="#f3f4f6",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        c900="#272727",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    button_primary_background_fill="#06AE56",
    button_primary_background_fill_dark="#06AE56",
    button_primary_background_fill_hover="#07C863",
    button_primary_border_color="#06AE56",
    button_primary_border_color_dark="#06AE56",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#F2F2F2",
    button_secondary_background_fill_dark="#2B2B2B",
    button_secondary_text_color="#393939",
    button_secondary_text_color_dark="#FFFFFF",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    block_title_text_color="*primary_500",
    block_title_background_fill="*primary_100",
    input_background_fill="#F6F6F6",
)
