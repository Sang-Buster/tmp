"""
LLM Agent with MCP-style tool calling for vehicle simulation.

Implements a proper tool calling loop:
  1. User message + tool schemas sent to LLM
  2. LLM decides which tool(s) to call (or answers directly)
  3. Tool results fed back to LLM for final response
  4. Multi-step reasoning supported (up to 5 tool calls per turn)
"""
import json
import re
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

MAX_TOOL_ROUNDS = 5

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
- To call a tool, respond with ONLY a JSON object (no extra text):
  {{"tool": "tool_name", "args": {{"param1": value1, "param2": value2}}}}
- You can call ONE tool per response. After seeing the result, you may call another.
- If no tool is needed, respond with a direct text answer (2-4 sentences).
- Use numbers (not strings) for numeric parameters: {{"x": 10}} not {{"x": "10"}}.

CRITICAL RULES FOR ZONE OPERATIONS:
- To DELETE a specific zone: FIRST call list_jamming_zones or list_spoofing_zones to get the zone ID, THEN call delete_jamming_zone or delete_spoofing_zone with that ID.
- To DELETE ALL zones: call clear_all_jamming_zones or clear_all_spoofing_zones directly.
- To CREATE a zone: call add_jamming_zone or add_spoofing_zone with x, y coordinates.
- Zone IDs are auto-generated and look like "zone_a1b2c3d4" for jamming or "zone_1" for spoofing. Never guess IDs -- always list first.
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
        Extract a tool call JSON from the LLM response.

        Handles common LLM output variations:
        - Pure JSON response
        - JSON on a single line amid text
        - Multi-line JSON object
        - JSON inside markdown code blocks
        - Trailing commas and minor formatting issues
        """
        text = content.strip()

        # 1) Direct JSON parse of the whole response
        parsed = self._try_parse_tool_json(text)
        if parsed:
            return parsed

        # 2) Extract from markdown code blocks (```json ... ``` or ``` ... ```)
        code_block_re = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
        for match in code_block_re.finditer(text):
            parsed = self._try_parse_tool_json(match.group(1).strip())
            if parsed:
                return parsed

        # 3) Find JSON object spanning multiple lines using brace matching
        for match in re.finditer(r'\{', text):
            start = match.start()
            candidate = self._extract_balanced_braces(text, start)
            if candidate and "tool" in candidate:
                parsed = self._try_parse_tool_json(candidate)
                if parsed:
                    return parsed

        return None

    @staticmethod
    def _try_parse_tool_json(text: str) -> Optional[dict]:
        """Try to parse text as a tool-call JSON, tolerating minor issues."""
        if not text:
            return None
        # Fix trailing commas before closing braces/brackets
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "tool" in data:
                if "args" not in data:
                    data["args"] = {}
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
        """Extract a balanced {...} substring starting at the given index."""
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, min(start + 2000, len(text))):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
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
