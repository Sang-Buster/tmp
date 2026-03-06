"""
LLM Agent with MCP-style tool calling for vehicle simulation.

Implements a proper tool calling loop:
  1. User message + tool schemas sent to LLM
  2. LLM decides which tool(s) to call (or answers directly)
  3. Tool results fed back to LLM for final response
  4. Multi-step reasoning supported (up to 3 tool calls per turn)
"""
import json
from typing import Any, Optional

from ..config import (
    LLM_MODEL,
    MISSION_END,
    SIMULATION_API_URL,
    X_RANGE,
    Y_RANGE,
    Z_RANGE,
    async_chat_with_retry,
)
from ..rag import add_log, get_logs, get_telemetry_history
from .tools import TOOL_SCHEMAS, execute_tool, get_tool_schemas_text

MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = f"""You are an expert assistant for a 3D multi-vehicle swarm simulation.

SIMULATION CONTEXT:
- Vehicles navigate from start positions to destination {MISSION_END}
- Map bounds: X={X_RANGE}, Y={Y_RANGE}, Z={Z_RANGE}
- Vehicles use communication-aware formation control
- Jamming zones degrade communication; spoofing attacks inject fake data
- Cryptographic authentication (HMAC-SHA256, ChaCha20, AES-256-CTR) can counter spoofing

AVAILABLE TOOLS:
{get_tool_schemas_text()}

INSTRUCTIONS:
- To call a tool, respond with EXACTLY one JSON object on its own line:
  {{"tool": "tool_name", "args": {{"param1": value1, "param2": value2}}}}
- You can call ONE tool per response. After seeing the result, you may call another.
- If no tool is needed, respond with a direct text answer (2-4 sentences).
- Be specific with numbers and positions. Format coordinates as (x, y, z).
- When asked about status, ALWAYS call get_agent_status or get_simulation_status first.
"""


class LLMAgent:
    """LLM Agent with MCP-style tool calling."""

    def __init__(self):
        self.model = LLM_MODEL
        self.api_url = SIMULATION_API_URL

    async def answer(self, user_query: str) -> dict:
        """
        Process a user message with tool calling loop.

        Returns a dict with 'response' (text) and 'tool_calls' (list).
        """
        print(f"\n{'='*60}")
        print(f"[LLM] Processing: {user_query}")

        add_log(user_query, source="user", message_type="command")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        tool_calls_made = []
        answer = "I was unable to generate a response. Is the LLM running?"
        last_content = ""

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            response = await async_chat_with_retry(self.model, messages=messages)

            if not response:
                answer = "I could not reach the LLM. Please check that Ollama is running and the model is loaded."
                break

            last_content = response["message"]["content"].strip()
            print(f"[LLM] Round {round_num}: {last_content[:120]}...")

            # Try to parse a tool call from the response
            tool_call = self._parse_tool_call(last_content)

            if tool_call is None:
                # No tool call -- this is the final answer
                answer = last_content
                break

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})
            print(f"[LLM] Tool call: {tool_name}({tool_args})")

            # Execute the tool
            result = await execute_tool(tool_name, tool_args)
            result_text = json.dumps(result, indent=2, default=str)

            tool_calls_made.append({
                "tool": tool_name,
                "args": tool_args,
                "result_summary": result.get("message", result.get("success", ""))
            })

            # Feed result back to LLM
            messages.append({"role": "assistant", "content": last_content})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n```json\n{result_text}\n```\n\nNow respond to the user based on this result. If you need more information, call another tool. Otherwise, give a concise final answer.",
            })
        else:
            # Loop completed all rounds without a break - use last response
            answer = last_content if last_content else "Reached maximum tool call rounds without a final answer."

        add_log(answer, source="llm", message_type="response")

        # Build response with tool call info
        if tool_calls_made:
            tools_summary = " | ".join(
                f"[{tc['tool']}]" for tc in tool_calls_made
            )
            print(f"[LLM] Tools used: {tools_summary}")

        print(f"[LLM] Answer ready ({len(answer)} chars)")
        print(f"{'='*60}\n")

        return {"response": answer, "tool_calls": tool_calls_made}

    def _parse_tool_call(self, content: str) -> Optional[dict]:
        """
        Try to extract a tool call JSON from the LLM response.
        Looks for {"tool": "...", "args": {...}} pattern.
        """
        # Try direct JSON parse
        try:
            data = json.loads(content.strip())
            if isinstance(data, dict) and "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("{") and "tool" in line:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "tool" in data:
                        return data
                except json.JSONDecodeError:
                    continue

        # Try extracting from code blocks
        if "```" in content:
            parts = content.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i].strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and "tool" in data:
                        return data
                except json.JSONDecodeError:
                    continue

        return None


# Global instance
_agent: Optional[LLMAgent] = None


def get_agent() -> LLMAgent:
    global _agent
    if _agent is None:
        _agent = LLMAgent()
    return _agent


async def answer_question(query: str) -> dict:
    agent = get_agent()
    return await agent.answer(query)
