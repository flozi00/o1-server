import json
import os
import time
from typing import List, Optional

import aiofiles
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

SYS_MSG = """ANALYSIS AND PROBLEM-SOLVING FRAMEWORK

1. TASK CLASSIFICATION
    Determine if the task is:
    a) Logical/Analytical (requires systematic problem-solving)
    b) Creative/Open-ended (requires imagination and originality)

2. FOR LOGICAL/ANALYTICAL TASKS:

    A. PROBLEM DECOMPOSITION
        - State the exact problem in one clear sentence
        - List all given information and constraints
        - Identify any missing information or assumptions needed
        - Define the expected output format

    B. COMPONENT ANALYSIS
        1. Primary Components:
            - List all main elements that directly affect the solution
            - Define their exact properties, units, and ranges
            - Note any constraints or limitations

        2. Secondary Components:
            - List supporting elements or factors
            - Define their influence on primary components
            - Document any dependencies

        3. Implicit Components:
            - Identify hidden assumptions
            - List unstated but necessary conditions
            - Document edge cases to consider

    C. RELATIONSHIP MAPPING
        1. Direct Relationships (1:1):
            - Document exact cause-effect pairs
            - List mathematical relationships
            - Define input-output correlations

        2. Indirect Relationships (1:n):
            - Map cascade effects
            - Document ripple impacts
            - List dependent variables

        3. Hierarchical Relationships:
            - Create parent-child structure
            - Define inheritance patterns
            - Document priority order

        4. Causal Relationships:
            - Map cause-effect chains
            - Document triggers and results
            - Define feedback loops

        5. Temporal Relationships:
            - Document time-dependent factors
            - List sequence requirements
            - Define timing constraints

    D. COMPREHENSION LEVELS
        1. Literal Understanding:
            - Basic facts and given information
            - Explicit statements
            - Direct requirements

        2. Contextual Understanding:
            - Domain-specific knowledge
            - Environmental factors
            - Situational constraints

        3. Abstract Understanding:
            - Pattern recognition
            - Underlying principles
            - General concepts

    E. SOLUTION DEVELOPMENT
        1. Assumptions:
            - List all assumptions explicitly
            - Justify each assumption
            - Document impact on solution

        2. Solution Steps:
            - Number each step
            - Provide detailed sub-steps
            - Include validation points

        3. Boundary Conditions:
            - Define valid input ranges
            - List exception cases
            - Document error handling

    F. SOLUTION VERIFICATION
        1. Logical Consistency:
            - Check step connections
            - Verify mathematical operations
            - Confirm logical flow

        2. Completeness:
            - Check all requirements met
            - Verify all cases handled
            - Confirm no missing steps

        3. Edge Cases:
            - Test minimum values
            - Test maximum values
            - Test boundary conditions

        4. Contradictions:
            - Check for logical conflicts
            - Verify consistent assumptions
            - Confirm no paradoxes

    USE TAGS:
    - Wrap each step explanation in <step> </step>
    - Wrap final solution in <answer> </answer>

3. FOR CREATIVE TASKS:
    - Focus on originality and innovation
    - Provide clear reasoning for creative choices
    - Wrap solution in <answer> </answer>

FEEDBACK SYSTEM:
    - User will provide feedback using <feedback> </feedback>
    - Use feedback to improve subsequent steps
    - Adjust approach based on feedback received

You are only allowed to generated one step at a time. Wait for the feedback before generating the next step.
Each step should be a short as possible. Make many small steps instead of a few large steps. One step should only contain a single idea.
If the task has a time based factor draw the states step by step. This needs to be done for example when solving leaderboards.

Examples:
<step> To find all multiples of 3 that are less than 10, we need to multiply 3 until its larger than 10.</step>
<step> Then we start with 3.</step>
<step> Then we add 3 to 3.</step>
<step> 3 + 3 = 6</step>
<step> Then we add 3 to 6.</step>
<step> 6 + 3 = 9</step>
<step> Then we add 3 to 9.</step>
<step> 9 + 3 = 12</step>
<step> Since 12 is larger than 10, we are done.</step>
<answer>3, 6, 9</answer>
"""

REWARDER = """You are an logical classifier.
The user will provide an question and steps of how to solve the question.
Your task is to critical rate the step of the user.
Answer with <feedback> </feedback> tags.

If the step is correct, write "Correct".
If the step is wrong, write "Wrong".
If the step is more than one fact at a time, write "Too much information in one step. Split this up into smaller parts and think again".

You are only allowed to answer with one of the three options. No more comments are allowed.

Examples:

<step> Um alle vielfachen von 3 die kleiner als 10 sind zu finden, muss man 3, 6 und 9 addieren. </step>
<feedback> Too much information in one step. Split this up into smaller parts and think again </feedback>
<step> Um alle vielfachen von 3 die kleiner als 10 sind zu finden muss man 3 so lange addieren bis man größer als 10 ist. </step>
<feedback> Correct </feedback>
<step> Dann beginnen wir mit 3 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 3 </step>
<feedback> Correct </feedback>
<step> 3 + 3 = 6 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 6 </step>
<feedback> Correct </feedback>
<step> 6 + 3 = 9 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 9 </step>
<feedback> Correct </feedback>
<step> 9 + 3 = 12 </step>
<feedback> Correct </feedback>
<step> Da 12 größer als 10 ist, sind wir fertig. </step>
<feedback> Correct </feedback>
<answer> 3, 6, 9 </answer>

The <step> tags are the steps of the user. The <feedback> tags are the feedback of the rewarder which is your job! The <answer> tags are the answer of the user. Your are not allowed to use the <answer> tags or even the <step> tags. Only the <feedback> tags are allowed.
Dont reply with any other tags than the <feedback> tags and its content.
"""
from dotenv import load_dotenv

load_dotenv()


import hashlib
import os

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=os.getenv("OAI_BASE_URL"),
    api_key=os.getenv("OAI_KEY"),
)

MODEL_TO_USE = os.getenv("LLM_NAME")

try:
    os.mkdir("logging")
except FileExistsError:
    pass


async def think_about(messages_given):
    msg_hash = hashlib.md5(json.dumps(messages_given).encode()).hexdigest()
    step_list = []
    reward_list = []
    full_answer = ""

    x = 0
    while x < 50:
        messages = [
            {"role": "system", "content": SYS_MSG},
        ]
        messages.extend(messages_given)

        for i in range(len(step_list)):
            messages.extend(
                [
                    {"role": "assistant", "content": step_list[i]},
                    {"role": "user", "content": reward_list[i]},
                ]
            )

        messages_reward = [
            {"role": "system", "content": REWARDER},
        ]
        messages_reward.extend(messages_given)

        for i in range(len(step_list)):
            messages_reward.extend(
                [
                    {"role": "user", "content": step_list[i]},
                    {"role": "assistant", "content": reward_list[i]},
                ]
            )

        completion = await client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        step = completion.choices[0].message.content
        step_list.append(step)

        completion = await client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=messages_reward,
            temperature=0.1,
        )
        reward = completion.choices[0].message.content
        reward_list.append(reward)

        if "<feedback>" not in reward:
            messages_reward.append(
                {
                    "role": "system",
                    "content": "Please provide feedback using <feedback> </feedback> tags.",
                }
            )
            completion = await client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=messages_reward,
                temperature=0.1,
            )
            reward = completion.choices[0].message.content
            reward_list[-1] = reward

        x += 1
        if "Correct" in reward:
            full_answer += step + "\n"
            yield step

        if "<answer>" in step:
            messages_given.append({"role": "assistant", "content": full_answer})
            async with aiofiles.open(f"logging/{msg_hash}.json", "w+") as f:
                await f.write(
                    json.dumps(
                        {
                            "steps": step_list,
                            "rewards": reward_list,
                            "prompt": messages_given,
                        }
                    )
                )
            break


app = FastAPI(title="OpenAI-compatible API")


# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


async def _resp_async_generator(messages: list):
    i = 0
    async for token in think_about(messages):
        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": "mock-gpt-model",
            "choices": [{"delta": {"content": token + " "}}],
        }
        i += 1
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        message_json = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
    if request.stream:
        return StreamingResponse(
            _resp_async_generator(message_json), media_type="application/x-ndjson"
        )

    answers = think_about(message_json)
    answer = ""
    async for a in answers:
        answer += a + " "
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=answer)}],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
