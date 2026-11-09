
import os
import json

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionToolParam

from agents.prompts import SYSTEM_PROMPT

from tools.data_analysis import (
    get_total_sales,
    get_average_sales,
    get_maximum_sales,
    get_product_sales,
    get_dataset_overview,
    get_missing_values
)


# ==========================================
# 1. CONNECT GROQ
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found!")

client = Groq(api_key=api_key)


# ==========================================
# 2. DEFINE TOOLS FOR GROQ
# ==========================================

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_total_sales",
            "description": "Calculate total sales in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_average_sales",
            "description": "Calculate average sales in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maximum_sales",
            "description": "Find the highest sales value.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_sales",
            "description": "Calculate and rank total sales by product.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_overview",
            "description": "Get dataset rows, columns and data types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_missing_values",
            "description": "Count missing values in each column.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ==========================================
# 3. MAP AI TOOL NAMES TO PYTHON FUNCTIONS
# ==========================================

TOOL_FUNCTIONS = {
    "get_total_sales": get_total_sales,
    "get_average_sales": get_average_sales,
    "get_maximum_sales": get_maximum_sales,
    "get_product_sales": get_product_sales,
    "get_dataset_overview": get_dataset_overview,
    "get_missing_values": get_missing_values
}


# ==========================================
# 4. EXECUTE SELECTED PYTHON FUNCTION
# ==========================================

def execute_tool(tool_name, arguments=None, df=None):

    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}")

    function = TOOL_FUNCTIONS[tool_name]

    arguments = arguments or {}

    return function(df=df, **arguments)


# ==========================================
# 5. AI AGENT TOOL CALLING LOOP
# ==========================================

def ask_agent(question, df=None):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # Return final answer if no tool is requested

        if not assistant_message.tool_calls:
            return assistant_message.content

        # Save assistant tool call

        messages.append(assistant_message)

        # Execute requested tools

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments or "{}"
            )

            result = execute_tool(
                tool_name,
                arguments,
                df=df
            )

            # Send result back to Groq

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

    return "Maximum tool-calling steps reached."


# ==========================================
# 6. RUN FROM TERMINAL
# ==========================================

if __name__ == "__main__":

    print("AI DATA ANALYST AGENT")
    print("--------------------")

    question = input("\nAsk your sales data question: ")

    answer = ask_agent(question)

    print("\nAI AGENT ANSWER:")
    print(answer)
