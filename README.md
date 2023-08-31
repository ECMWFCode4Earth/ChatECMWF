# 🚀 ChatECMWF: exploring ECMWF resources and datasets using natural language queries

## Introduction

The [European Centre for Medium-Range Weather Forecasts (ECMWF)](https://www.ecmwf.int) has a very large amount of data, in the order of [hundreds of petabytes](https://www.ecmwf.int/en/about/media-centre/key-facts-and-figures). The access to the data is granted to users via a few different APIs, as well as through web interfaces. Due to the complexity of the data stored — thousand of different metereological variables — the discoverability of the data is not ideal, and a new users might feel overwhelmed by the large amount of data-retrieval options available to her. ECMWF documentation is also extensive, and suffers similar from discoverability problems.

ChatECMWF aims at solving this problem via a chatbot, allowing a user to describe in natural language the information she is interested in. For instance a query could be

    I would like to see data on the hurricane Katrina.

or

    Did it rain in North-East Italy the last week of June 1987?

or

    How can I download data from the CDSAPI? How is the .cdsapirc file looking? 


ChatECMWF interacts with the user in a conversational style, asking for more details when appropriate, and finally answering with either a (possibly interactive) plot of the data, or with a textual answer based on data it retrieved from ECMWF Confluence [knowledge base](https://confluence.ecmwf.int) or github repositories.

## Project outputs

The main output of the project is a chatbot, provided for convenience in Dockerized containers. [Gradio](https://www.gradio.app/) has been employed for building the interface. Currently, the LLM is not answering asynchronously, but it should be trivial implementing the asynchronous methods for the LLM agent within Langchain.

The conversational Q&A tool requires a vector database generated from ECMWF's Confluence knowledge base, which is also provided as part of the project, along with the scripts to re-generate it if needed.

Finally, the [tests](/tests) folder contains simple functional tests which check whether the responses of the bot integrates successfully with the gradio dashboard.

## How to use the software

To ensure a seamless experience in any type of infrastructure, we employ [Docker](https://www.docker.com) containers to run the software in a reproducible way. Therefore, the requirements to run chatECMWF is ```docker```, together with the needed API keys (OpenAI and HuggingFaceHub) for employing the LLMs.

> **_NOTE:_**  Employing the main branch requires an OpenAI API Key for ChatGPT, which, in the present time, is the best choice for a stable and seamless experience. ChatGPT credits pricing is explained in details [here](https://openai.com/pricing); you will not need many of them as a single user, but it would be better to estimate the costs if you're planning to use them in production for many users. A free alternative, based on [Replicate](https://replicate.com/) base credits, is available in the ```llama2``` branch of this repository.

Please follow these steps:

1. Clone the repository via the command ```git clone https://github.com/ECMWFCode4Earth/ChatECMWF.git```

2. ```cd chatECMWFinterface```

3. Setup the required environment variables (```OPENAI_API_KEY``` and ```HUGGINGFACEHUB_API_TOKEN```) in the [.env](.env) file. Several API keys are required, in particular those from [OpenAI](https://openai.com/blog/openai-api), [HuggingFace](https://huggingface.co/docs/huggingface_hub/v0.5.1/en/package_reference/hf_api). You can also customize the desired version and temperature of the GPT model, setting the ```GPT_VERSION``` and ```GPT_TEMPERATURE```, as well as ```MAX_TOKENS```, which can control the length of generated texts. All the possible configurations can be found in the [src/config.py](src/config.py) file.

4. Setup the CDS API key in the [.cdsapirc](.cdsapirc) file, following the [documentation](https://cds.climate.copernicus.eu/api-how-to).

5. ```mkdir vector_db```

6. Unzip the saved databases [openapi](https://sciscry-my.sharepoint.com/:u:/g/personal/piero_sciscry_ai/Eb_QXGOxnxdCuqmgqdX2AoYBE0i-JiNh9TRNTRi1cjd44Q?e=nBTqBt), [web](https://sciscry-my.sharepoint.com/:u:/g/personal/piero_sciscry_ai/EWjSkmq3BdZKr49H4-1DU3gB0a0_Jupxz3qOpPwnek3efw?e=gLnQI3) and [confluence and github](https://sciscry-my.sharepoint.com/:u:/g/personal/piero_sciscry_ai/EUsfeotdlqFJlyrhO2uj87wBL_aoBzi8UWUd4VrW30ys2Q?e=vdKIZL) into the ```vector_db``` folder.

7. Run the bash script ```run.sh```

8. The chatbot is now available at [http://127.0.0.1:8000](http://127.0.0.1:8000)


## Methodology

The most important technologies powering ChatECMWF are [Langchain](https://www.langchain.com), a framework for developing applications powered by LLMs, and [Deeplake](https://www.deeplake.ai), a vector database allowing to find match the user query to a corpus of data based on semantic, rather than merely lexicographic, similarity.

### Langchain: automated LLM workflows for retrieving complex information

The theoretical base for this project relies in the [ReAct](https://arxiv.org/abs/2210.03629) framework. ReAct stands for Reasoning and Acting: the model plans actions through a chain-of-thoughts process, i.e. the LLM asks question to itself, making observations, understanding which information are missing to complete the task, and then act employing external resources, as unstructured data persisted in vector databases, or APIs and search engines. 
Langchain is a framework for building an AI agent, based on a chat-tuned LLM, such as [ChatGPT](https://chat.openai.com) or the chat-tuned versions of [Llama2](https://ai.meta.com/llama/). Langchain enables the creations of different workflows that include many different API calls, request for clarifications to the user, queries to the LLM. These workflows result in a complex interaction with the user, which far exceed the original LLM capabilities. Think for instance of a personal assistant Langchain application: after being informed of a meeting, it might decide to book a room using a specify API, check for conflicts in the schedule using another API, ask for clarifications if some details of the meeting are not specified in the original.

One key concept in a Langchain is the concept of tool. Tools are interfaces that an agent can use to interact with the world. We defined different tools, each one corresponding to a different resource provided by ECMWF:

- ECMWF web and Confluence content, resummarized directly from ChatGPT
- Source code from the main github repositories in the ECMWF official space
- [Meteogram API](https://www.ecmwf.int/sites/default/files/elibrary/2017/17307-eccharts-and-web-services-update.pdf)
- [Charts API](https://charts.ecmwf.int)
- the reanalsys at single-pressure levels from the [CDS API](https://cds.climate.copernicus.eu/#!/home) together with a utility to plot the downloaded grib files

As the user interacts with the AI agent, the queries are analyzed, the agent will ask for clarifications if needed, and it will pass the request to the appropriate tool. A tool can, in turn, provide an answer, or ask for further clarification.

### Vector databases and semantic search

The `Documentation' tool makes use of a vector database to match a user query to corresponding pages in the ECMWF Confluence knowledge base.

We make use of [Sentence Transformers](https://huggingface.co/sentence-transformers) to convert sentences and paragraphs of text into a high-dimensional vectors; a Sentence Transformer will generate vectors that are close to each other if the meaning of the sentences is similar, transforming semantic similarity to geometric similarity.

The whole ECMWF Confluence knowledge bases in split into chunks of pre-determined size, and each chunk is associated to a vector using a Sentence Transformer. The user query is also associated to a vector. In order to reply to the user query we select the 5 closest vectors, according to the Euclidean norm, and we ask the LLM to reply to a prompt such as

    Reply to the following question:
    
    <USER QUERY>
    
    using the following information
    
    <RELEVANT INFORMATION 1>
    ...
    <RELEVANT INFORMATION 5>

There are a few advantages in this approach: first of all, the search is semantic, meaning that for example `rain` and `precipitation` are assigned to very similar vectors. Furthermore, the approach is very scalable, as vector database can run similarity searches efficiently even on a very large number of vectors. Finally, the approach is flexible and easily extensibile, as vector databases can be used

## Testing

In the ``tests/`` folder we maintain a list of test questions. These serve mainly for functional testing, i.e. that the model and tools developed return a final answer, and no internal error or exception is raised. Proper evaluation of the LLM answers quality should be performed through a benchmarked series of questions and answers reviewed from humans. 
The evaluation of the quality of natural-language answers to natural-language queries is complex and in general difficult to automatize.

For this reason we only test for the completion of the workflow spawned by a question, but we do not assess the quality of the answer itself. In the future, it could be possible to have an LLM (such as GPT-4) rate the quality of the answers, or to crowd-source the rating process, see for instance [here](https://chat.lmsys.org/?arena).

## Authors

[Piero Ferrarese](mailto:piero@sciscry.ai)  

[Giacomo Bighin](mailto:bighin@gmail.com)

## Mentors

[Baudouin Raoult]()

[Sylvie Lamy-Thepaut]()

[Helen Setchell]()

[Myranda Uselton Shirk]()

## License

    Copyright 2023, European Union.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
