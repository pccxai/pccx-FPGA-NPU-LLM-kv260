const statusEl = document.querySelector("#status");
const messagesEl = document.querySelector("#messages");
const composer = document.querySelector("#composer");
const promptEl = document.querySelector("#prompt");
const sendEl = document.querySelector("#send");
const resetEl = document.querySelector("#reset");

const sessionId = crypto.randomUUID
  ? crypto.randomUUID()
  : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

let ws;
let activeAssistant;

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/api/chat`);
  ws.addEventListener("open", () => {
    statusEl.textContent = "Connected";
  });
  ws.addEventListener("close", () => {
    statusEl.textContent = "Disconnected, retrying";
    setTimeout(connect, 1200);
  });
  ws.addEventListener("error", () => {
    statusEl.textContent = "Connection error";
  });
  ws.addEventListener("message", (event) => {
    const frame = JSON.parse(event.data);
    if (frame.type === "token") {
      if (!activeAssistant) {
        activeAssistant = appendMessage("assistant", "");
      }
      activeAssistant.textContent += frame.content;
      scrollToEnd();
      return;
    }
    if (frame.type === "done") {
      activeAssistant = null;
      sendEl.disabled = false;
      promptEl.focus();
      statusEl.textContent = `Done, ${frame.tokens} tokens`;
      return;
    }
    if (frame.type === "error") {
      activeAssistant = null;
      sendEl.disabled = false;
      appendMessage("error", frame.message || "Request failed");
    }
  });
}

function appendMessage(kind, text) {
  const node = document.createElement("div");
  node.className = `msg ${kind}`;
  node.textContent = text;
  messagesEl.appendChild(node);
  scrollToEnd();
  return node;
}

function scrollToEnd() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = promptEl.value.trim();
  if (!content || !ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  appendMessage("user", content);
  activeAssistant = appendMessage("assistant", "");
  sendEl.disabled = true;
  statusEl.textContent = "Generating";
  promptEl.value = "";
  ws.send(JSON.stringify({
    type: "user_message",
    content,
    session_id: sessionId,
    temperature: 0.7,
    top_p: 0.95,
    max_new_tokens: 128
  }));
});

resetEl.addEventListener("click", () => {
  messagesEl.replaceChildren();
  activeAssistant = null;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type: "reset", session_id: sessionId}));
  }
});

connect();
