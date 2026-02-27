/**
 * Chat interface with MCP tool-calling support
 */

const Chat = {
  messagesContainer: null,
  input: null,
  sendButton: null,

  init() {
    this.messagesContainer = document.getElementById("chat-messages");
    this.input = document.getElementById("chat-input");
    this.sendButton = document.getElementById("chat-send");

    this.sendButton.addEventListener("click", () => this.sendMessage());
    this.input.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    this.addMessage(
      "system",
      'MCP Tools active. Try "move agent1 to 5, 5", "add a phantom spoofing zone at 20, 60", "enable crypto auth", or ask "where are the agents?"',
    );
  },

  addMessage(role, content) {
    const msg = document.createElement("div");

    const baseClasses = "text-sm rounded-lg px-3 py-2 max-w-[90%]";
    if (role === "user") {
      msg.className = `${baseClasses} bg-info/20 text-foreground ml-auto`;
    } else if (role === "assistant") {
      msg.className = `${baseClasses} bg-secondary text-foreground`;
    } else if (role === "tool") {
      msg.className = `${baseClasses} bg-green-500/10 text-green-200 border border-green-500/30 text-xs font-mono`;
    } else {
      msg.className = `${baseClasses} bg-muted/50 text-muted-foreground text-xs italic`;
    }

    // Support basic markdown-like formatting
    if (typeof content === "string") {
      msg.innerHTML = this._formatContent(content);
    } else {
      msg.textContent = content;
    }
    this.messagesContainer.appendChild(msg);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  },

  _formatContent(text) {
    return text
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.*?)`/g, '<code class="bg-secondary/50 px-1 rounded text-xs">$1</code>')
      .replace(/\n/g, "<br>");
  },

  addLLMMessage(agentId, reasoning, direction) {
    const msg = document.createElement("div");
    msg.className =
      "text-xs rounded-lg px-3 py-2 max-w-[90%] bg-purple-500/20 text-purple-200 border border-purple-500/30";

    const dirStr = direction
      ? ` → [${direction.map((d) => d.toFixed(2)).join(", ")}]`
      : "";

    msg.innerHTML = `
      <div class="flex items-center gap-1.5 mb-1 text-purple-300 font-medium">
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3 4 4.5 4.5 0 0 1-3-4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>
        LLM Assisting ${agentId}${dirStr}
      </div>
      <div class="text-purple-100/80">${reasoning}</div>
    `;

    this.messagesContainer.appendChild(msg);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  },

  _seenLLMActivity: new Set(),

  processLLMActivity(activity) {
    if (!activity || activity.length === 0) return;

    for (const entry of activity) {
      const key = `${entry.agent_id}_${entry.timestamp}`;
      if (!this._seenLLMActivity.has(key)) {
        this._seenLLMActivity.add(key);
        this.addLLMMessage(entry.agent_id, entry.reasoning, entry.direction);

        if (this._seenLLMActivity.size > 100) {
          const oldest = this._seenLLMActivity.values().next().value;
          this._seenLLMActivity.delete(oldest);
        }
      }
    }
  },

  async sendMessage() {
    const message = this.input.value.trim();
    if (!message) return;

    this.addMessage("user", message);
    this.input.value = "";

    this.input.disabled = true;
    this.sendButton.disabled = true;

    // Show thinking indicator
    const thinkingEl = document.createElement("div");
    thinkingEl.className =
      "text-xs rounded-lg px-3 py-2 max-w-[90%] bg-secondary/50 text-muted-foreground animate-pulse";
    thinkingEl.textContent = "Thinking...";
    thinkingEl.id = "chat-thinking";
    this.messagesContainer.appendChild(thinkingEl);
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      // Remove thinking indicator
      const thinking = document.getElementById("chat-thinking");
      if (thinking) thinking.remove();

      const data = await response.json();

      if (data.tool_calls && data.tool_calls.length > 0) {
        for (const tc of data.tool_calls) {
          this.addMessage("tool", `⚡ ${tc.tool}(${JSON.stringify(tc.args)})`);
        }
      }

      this.addMessage("assistant", data.response || "No response");
    } catch (error) {
      const thinking = document.getElementById("chat-thinking");
      if (thinking) thinking.remove();
      this.addMessage("system", `Error: ${error.message}`);
    } finally {
      this.input.disabled = false;
      this.sendButton.disabled = false;
      this.input.focus();
    }
  },
};

window.Chat = Chat;
