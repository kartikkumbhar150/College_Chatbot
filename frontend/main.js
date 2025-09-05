const apiBase = "/";
let listening = false;
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const convEl = document.getElementById("conversation");
const btn = document.getElementById("toggle-listen");
const textInput = document.getElementById("text-input");
const sendBtn = document.getElementById("send-btn");
const langSelect = document.getElementById("lang-select");
const sessionId = "session-" + Math.random().toString(36).slice(2, 9);

let finalTranscript = "";
let awake = false;
let ttsInterrupted = false;
let stopRequested = false;

function containsWake(text) {
  return /\b(hey dit|hello dit|hello|hi|hi dit)\b/i.test(text);
}
function containsStop(text) {
  return /\b(stop|okay stop|ok stop|wait|exit)\b/i.test(text);
}

function stopSpeaking() {
  window.speechSynthesis.cancel();
  ttsInterrupted = true;
  stopRequested = true;
  awake = true;
  statusEl.textContent = "Stopped. Ask me another question...";
}

let recognition;
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = langSelect?.value || "en-IN";

  recognition.onstart = () => {
    statusEl.textContent = "Listening. Say hello to wake.";
  };

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript = event.results[i][0].transcript.trim();
        transcriptEl.textContent = "Heard: " + finalTranscript;
        handleFinalSpeech(finalTranscript.toLowerCase());
      } else {
        interim += event.results[i][0].transcript;
        transcriptEl.textContent = "Listening: " + interim;
      }
    }
  };

  recognition.onerror = () => {
    statusEl.textContent = "Speech recognition error.";
  };

  recognition.onend = () => {
    if (listening) recognition.start();
  };
} else {
  statusEl.textContent = "Speech Recognition not supported.";
}

if (langSelect) {
  langSelect.addEventListener("change", () => {
    if (recognition) recognition.lang = langSelect.value;
  });
}

async function translateToEnglish(text, sourceLang = "auto") {
  try {
    if (sourceLang && sourceLang.includes("-")) {
      sourceLang = sourceLang.split("-")[0];
    }
    const res = await fetch("https://libretranslate.de/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: text,
        source: sourceLang || "auto",
        target: "en",
        format: "text"
      }),
    });
    const data = await res.json();
    return data.translatedText || text;
  } catch {
    return text;
  }
}

async function handleFinalSpeech(text) {
  if (containsStop(text)) {
    stopSpeaking();
    return;
  }
  if (!awake) {
    if (containsWake(text)) {
      stopRequested = false;
      awake = true;
      statusEl.textContent = "Awake. Ask your question...";
      appendConversation("Yes, how can I help?", "bot");
      speak("Yes, how can I help?");
    }
  } else {
    statusEl.textContent = "Processing...";
    appendConversation(text, "user");
    const englishText = await translateToEnglish(text, recognition.lang);
    handleBotResponse(sendQuery(englishText));
  }
}

async function sendQuery(q) {
  const res = await fetch(apiBase + "api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q, session_id: sessionId }),
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { error: text || `Server returned ${res.status}` };
  }
  if (!res.ok) {
    throw new Error(body.detail || body.error || `Server returned ${res.status}`);
  }
  return body;
}

function appendConversation(text, sender = "bot") {
  const d = document.createElement("div");
  d.classList.add("message", sender);
  if (sender === "bot") {
    try {
      const html = DOMPurify.sanitize(marked.parse(text || ""), {
        USE_PROFILES: { html: true },
      });
      d.innerHTML = html;
    } catch {
      d.textContent = text;
    }
  } else {
    d.textContent = text;
  }
  convEl.appendChild(d);
  convEl.scrollTop = convEl.scrollHeight;
}

let voices = [];
window.speechSynthesis.onvoiceschanged = () => {
  voices = window.speechSynthesis.getVoices();
};
function speak(text) {
  if (!text) return;
  if (containsStop(text)) return;
  let cleanText = text
    .replace(/#+\s*/g, "")
    .replace(/[-*]\s+/g, "")
    .replace(/^\d+\.\s+/gm, "");
  window.speechSynthesis.cancel();
  ttsInterrupted = false;
  const parts = cleanText.split(/(?<=[.!?])\s+/).filter(Boolean);
  const speakNext = () => {
    if (ttsInterrupted || parts.length === 0) return;
    const u = new SpeechSynthesisUtterance(parts.shift());
    const prefer = voices.find(
      (v) => /india|en-in|hindi/i.test(v.name) || /en-?in/i.test(v.lang)
    );
    if (prefer) u.voice = prefer;
    u.lang = langSelect?.value || "en-IN";
    u.onstart = () => (statusEl.textContent = "Speaking...");
    u.onend = () => {
      if (!window.speechSynthesis.speaking) {
        statusEl.textContent = "Ready for next question...";
      }
      if (!ttsInterrupted) speakNext();
    };
    window.speechSynthesis.speak(u);
  };
  speakNext();
}

function showTyping() {
  const typing = document.createElement("div");
  typing.classList.add("typing");
  typing.innerHTML = "<span></span><span></span><span></span>";
  convEl.appendChild(typing);
  convEl.scrollTop = convEl.scrollHeight;
  return typing;
}
function handleBotResponse(promise) {
  const typing = showTyping();
  promise
    .then((resp) => {
      typing.remove();
      if (stopRequested) {
        statusEl.textContent = "Stopped. Waiting for hello...";
        return;
      }
      appendConversation(resp.answer, "bot");
      speak(resp.answer);
      awake = true;
      statusEl.textContent = "Ready for next question...";
    })
    .catch((err) => {
      typing.remove();
      if (!stopRequested) {
        statusEl.textContent = "Error contacting backend: " + err.message;
      }
    });
}

btn.addEventListener("click", () => {
  listening = !listening;
  if (listening && recognition) {
    recognition.start();
    btn.textContent = "Stop Listening";
  } else {
    if (recognition) recognition.stop();
    btn.textContent = "Start Listening";
    statusEl.textContent = "Stopped.";
  }
});

async function handleTextSubmit() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  statusEl.textContent = "Processing...";
  appendConversation(text, "user");
  const englishText = await translateToEnglish(text, recognition.lang);
  handleBotResponse(sendQuery(englishText));
}
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleTextSubmit();
});
sendBtn.addEventListener("click", handleTextSubmit);
