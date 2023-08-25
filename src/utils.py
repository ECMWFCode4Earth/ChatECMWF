"""
Some utility functions for improving formatting of the output on the Gradio dashboard.
"""
from typing import Any, Dict

import numpy as np


def sources_markdown(sources: Dict[str, Any], source_field: str) -> str:
    """
    This function takes as input the output of a RetrievalQA tool from langchain,
    and properly format the links. As the original metadata in the vector DB were saved ill-formed,
    a custom logic has been implemented for recovering the full URL.

    Github links has a '+' in place of spaces; Confluence links are saved with underscores in place of dashes.

    Args:
        sources: Dict[str, Any]
        source_field: str
                 Either 'source' or 'link'
    Returns:
        output: str
                The formatted string containing the list of linkgs to the source documents
    """
    links = ""
    for source_doc in sources["source_documents"]:
        if source_field == "source":
            if "+" not in source_doc.metadata[source_field]:
                href_link = build_github_link(source_doc.metadata[source_field])
            else:
                link = source_doc.metadata[source_field]
                spaces = link.split("_")[:2]
                page = link.split("_")[-1].replace(".txt", "")
                href_link = (
                    f"https://confluence.ecmwf.int/{spaces[0]}/{spaces[1]}/{page}"
                )
        else:
            href_link = source_doc.metadata[source_field]
        link = '<a href="{1}" > {0} </a> <br/>'.format(
            *[source_doc.metadata[source_field].split("/")[-1], href_link]
        )
        links += link
    return f"<br/>{links}"


def build_github_link(simple_link: str, base_domain: str = "ecmwf") -> str:
    """
    A utility to rebuild a github link organized as base domain (in our case, 'ecmwf'), repository name, and file path.

    Args:
       simple_link: str
                    base_domain/repo_name/path/
       base_domain: str
    Returns:
       github_link: str
    """
    repo_name = simple_link.split("/")[0]
    path = "/".join(simple_link.split("/")[1:])
    github_link = f"http://github.com/{base_domain}/{repo_name}/blob/master/{path}"
    return github_link
