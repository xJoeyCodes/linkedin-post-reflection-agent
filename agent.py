import os
from typing import List, Annotated, TypedDict, Sequence

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; environment variables will still be read from the environment
    pass

from langchain_ibm import ChatWatsonx
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END, MessageGraph



generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts."
            " Generate the best LinkedIn post possible for the user's request."
            " If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),

    MessagesPlaceholder(variable_name="messages")
])


class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]


def init_llm():
    """Initialize an LLM client using environment variables.

    Required (for IBM provider): WATSONX_MODEL_ID, WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_API_KEY
    Copy `.env.example` to `.env` and fill values if you don't have environment variables set.
    """
    provider = os.getenv("MODEL_PROVIDER", "ibm").lower()
    if provider == "ibm":
        model_id = os.getenv("WATSONX_MODEL_ID")
        url = os.getenv("WATSONX_URL")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        api_key = os.getenv("WATSONX_API_KEY")
        if not all([model_id, url, project_id, api_key]):
            raise RuntimeError(
                "Missing IBM Watsonx environment variables.\n"
                "Create a .env file from .env.example and set WATSONX_MODEL_ID, WATSONX_URL, WATSONX_PROJECT_ID, WATSONX_API_KEY"
            )
        return ChatWatsonx(
            model_id=model_id,
            url=url,
            project_id=project_id,
            api_key=api_key,
        )
    else:
        raise RuntimeError(f"Unsupported MODEL_PROVIDER: {provider}")


def run_workflow(input_text: str):
    llm = init_llm()

    # build chains with the initialized LLM
    generate_chain = generation_prompt | llm
    reflect_chain = reflection_prompt | llm

    # Initialize a MessageGraph and wire nodes
    graph = MessageGraph()

    def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
        generated_post = generate_chain.invoke({"messages": state})
        content = getattr(generated_post, "content", None) or str(generated_post)
        return [AIMessage(content=content)]

    def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        res = reflect_chain.invoke({"messages": messages})
        content = getattr(res, "content", None) or str(res)
        return [HumanMessage(content=content)]

    graph.add_node("generate", generation_node)
    graph.add_node("reflect", reflection_node)
    graph.add_edge("reflect", "generate")
    graph.set_entry_point("generate")

    def should_continue(state: List[BaseMessage]):
        if len(state) > 6:
            return END
        return "reflect"

    graph.add_conditional_edges("generate", should_continue)

    workflow = graph.compile()
    inputs = HumanMessage(content=input_text)

    response = workflow.invoke(inputs)
    print(response)
    return response


if __name__ == "__main__":
    try:
        run_workflow(
            "Write a linkedin post on getting a software developer job at IBM under 160 characters"
        )
    except Exception as e:
        print("Error:", e)
        print("Tip: copy .env.example to .env and set environment variables.")



