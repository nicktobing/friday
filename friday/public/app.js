// Friday — the voice loop (runs entirely in the browser).
//
// Flow:  you speak  →  Web Speech API transcribes (free, on-device)
//        →  send transcript + history to /chat  →  Claude streams a reply
//        →  speechSynthesis speaks it out loud automatically (no file to tap)
//        →  Friday listens again.
//
// Tap the orb once to start a session (a user gesture is required by iOS to
// unlock the microphone and audio playback). Tap again to stop. Tap while
// Friday is talking to interrupt her (barge-in).

const orb = document.getElementById("orb");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

// Conversation history sent to the brain each turn (the API is stateless).
const history = [];

let active = false;       // is a session running?
let recognizing = false;  // is the recognizer currently listening?
let recognition = null;
let preferredVoice = null;

// ── Speaker identification (Picovoice Eagle — runs entirely in-browser) ────────
// Profiles stored in localStorage as [{name, profileBytes: base64}]

let currentSpeaker = null;
let eagleInstance = null;          // lazily created Eagle identifier
let captureCtx = null;             // AudioContext for PCM capture
let captureProcessor = null;       // ScriptProcessorNode
let captureBuffer = [];            // accumulated Float32 samples this utterance

function loadSpeakerProfiles() {
  try { return JSON.parse(localStorage.getItem("eagle_speakers") || "[]"); } catch (_) { return []; }
}

function b64ToUint8(b64) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

function resampleTo16k(float32, fromRate) {
  if (fromRate === 16000) return float32;
  const ratio = fromRate / 16000;
  const outLen = Math.floor(float32.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, float32.length - 1);
    const frac = srcIdx - lo;
    out[i] = float32[lo] * (1 - frac) + float32[hi] * frac;
  }
  return out;
}

async function getEagle() {
  if (eagleInstance) return eagleInstance;
  const accessKey = localStorage.getItem("picovoice_access_key");
  if (!accessKey || typeof EagleWeb === "undefined") return null;
  try {
    eagleInstance = await EagleWeb.Eagle.create(accessKey, { publicPath: "/eagle/eagle_params.pv" });
  } catch (e) {
    console.warn("Eagle init failed:", e.message);
  }
  return eagleInstance;
}

function startCapture() {
  captureBuffer = [];
  if (!captureCtx || !captureProcessor) return;
  captureProcessor.onaudioprocess = (e) => {
    const chunk = e.inputBuffer.getChannelData(0);
    captureBuffer.push(new Float32Array(chunk));
  };
}

function stopCapture() {
  if (captureProcessor) captureProcessor.onaudioprocess = null;
  if (!captureBuffer.length) return null;
  const total = captureBuffer.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of captureBuffer) { merged.set(chunk, offset); offset += chunk.length; }
  captureBuffer = [];
  return merged;
}

async function identifySpeaker() {
  const profiles = loadSpeakerProfiles();
  if (!profiles.length) return null;
  const eagle = await getEagle();
  if (!eagle) return null;

  const float32 = stopCapture();
  if (!float32 || float32.length < 1000) return null;

  const rate = captureCtx ? captureCtx.sampleRate : 44100;
  const resampled = resampleTo16k(float32, rate);
  if (resampled.length < eagle.minProcessSamples) return null;

  const pcm = new Int16Array(resampled.length);
  for (let i = 0; i < resampled.length; i++) {
    pcm[i] = Math.max(-32768, Math.min(32767, Math.round(resampled[i] * 32768)));
  }

  const eagleProfiles = profiles.map(p => ({ bytes: b64ToUint8(p.profileBytes) }));
  try {
    const scores = await eagle.process(pcm, eagleProfiles);
    const best = scores.indexOf(Math.max(...scores));
    if (scores[best] > 0.5) return profiles[best].name;
  } catch (e) {
    console.warn("Eagle identify error:", e.message);
  }
  return null;
}

async function initCaptureNode(stream) {
  try {
    captureCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (captureCtx.state === "suspended") await captureCtx.resume().catch(() => {});
    const source = captureCtx.createMediaStreamSource(stream);
    captureProcessor = captureCtx.createScriptProcessor(1024, 1, 1);
    source.connect(captureProcessor);
    captureProcessor.connect(captureCtx.destination);
  } catch (e) {
    console.warn("Capture node init failed:", e.message);
  }
}

// Show who's speaking in the status bar (briefly).
function showSpeaker(name) {
  if (!name) return;
  statusEl.textContent = `${name} — Listening…`;
  setTimeout(() => { if (statusEl.textContent === `${name} — Listening…`) statusEl.textContent = "Listening…"; }, 2000);
}

// ---------- UI helpers ----------
function setState(state, message) {
  document.body.className = state; // "", "listening", "thinking", "speaking"
  if (message !== undefined) statusEl.textContent = message;
}

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = "bubble " + (role === "user" ? "user" : "friday");
  div.textContent = text;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return div;
}

// ---------- Voice output ----------
// Two modes: "elevenlabs" (natural, via the server) or "browser" (free, on-device).
// The server tells us which at /config. Chunks are spoken in order through a queue
// so streamed sentences don't overlap.
let ttsMode = "browser";
let speakQueue = [];
let speaking = false;
// AudioContext approach for ElevenLabs: coexists with SpeechRecognition on iOS
// whereas <audio> elements fight the mic for the audio session.
let audioCtx = null;
let currentSource = null;

async function loadConfig() {
  try {
    const cfg = await (await fetch("/config")).json();
    ttsMode = cfg.tts || "browser";
  } catch (_) { /* keep browser default */ }
}
loadConfig();

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  if (!voices.length) return;
  preferredVoice =
    voices.find(v => /en[-_]?(GB|US)/i.test(v.lang) && /female|samantha|aria|jenny|libby/i.test(v.name)) ||
    voices.find(v => /^en/i.test(v.lang)) ||
    voices[0];
}
if ("speechSynthesis" in window) {
  pickVoice();
  speechSynthesis.onvoiceschanged = pickVoice;
}

// Queue a chunk of text to be spoken (non-blocking).
function enqueueSpeak(text) {
  if (!text.trim()) return;
  speakQueue.push(text);
  if (!speaking) drainSpeakQueue();
}

async function drainSpeakQueue() {
  speaking = true;
  while (speakQueue.length) {
    const text = speakQueue.shift();
    try {
      if (ttsMode === "elevenlabs") await speakEleven(text);
      else await speakBrowser(text);
    } catch (_) {
      try { await speakBrowser(text); } catch (_) {} // fall back if ElevenLabs fails
    }
  }
  speaking = false;
}

function speakBrowser(text) {
  return new Promise(resolve => {
    const u = new SpeechSynthesisUtterance(text);
    if (preferredVoice) u.voice = preferredVoice;
    u.rate = 1.0;
    u.pitch = 1.0;
    u.onend = resolve;
    u.onerror = resolve;
    speechSynthesis.speak(u);
  });
}

async function speakEleven(text) {
  const res = await fetch("/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("tts " + res.status);
  const arrayBuffer = await res.arrayBuffer();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);
  return new Promise(resolve => {
    if (currentSource) { try { currentSource.stop(); } catch (_) {} currentSource = null; }
    const source = audioCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(audioCtx.destination);
    source.onended = () => { currentSource = null; resolve(); };
    source.start();
    currentSource = source;
  });
}

// Stop everything immediately (barge-in / session end).
function stopSpeaking() {
  speakQueue = [];
  speaking = false;
  speechSynthesis.cancel();
  if (currentSource) { try { currentSource.stop(); } catch (_) {} currentSource = null; }
}

function isSpeaking() {
  return speaking || speakQueue.length > 0 || speechSynthesis.speaking || currentSource !== null;
}

// Unlock audio inside the user's tap gesture (required by iOS).
function unlockAudio() {
  try { speechSynthesis.speak(new SpeechSynthesisUtterance(" ")); } catch (_) {}
  // Create and unlock an AudioContext for ElevenLabs playback.
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  // Play a silent buffer so iOS marks this context as user-activated.
  try {
    const buf = audioCtx.createBuffer(1, 1, audioCtx.sampleRate);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start();
  } catch (_) {}
}

// ---------- Talking to the brain ----------
async function askFriday(userText, speakerName = null) {
  history.push({ role: "user", content: userText });
  setState("thinking", "Friday is thinking…");

  const reply = addBubble("friday", "");
  let full = "";
  let spokenUpTo = 0; // index in `full` we've already sent to TTS

  // Speak complete sentences as they stream in, for low latency.
  const flushSentences = (final = false) => {
    const pending = full.slice(spokenUpTo);
    const matches = [...pending.matchAll(/[^.!?]+[.!?]+(\s|$)/g)];
    let cut = 0;
    for (const m of matches) cut = m.index + m[0].length;
    const ready = final ? pending : pending.slice(0, cut);
    if (ready.trim()) {
      spokenUpTo += final ? pending.length : cut;
      enqueueSpeak(ready);
    }
  };

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, speakerName }),
    });

    const readerStream = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    setState("speaking", "Friday");

    while (true) {
      const { value, done } = await readerStream.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") continue;
        try {
          const obj = JSON.parse(data);
          if (obj.error) throw new Error(obj.error);
          if (obj.text) {
            full += obj.text;
            reply.textContent = full;
            transcriptEl.scrollTop = transcriptEl.scrollHeight;
            flushSentences(false);
          }
        } catch (e) { /* ignore partial JSON */ }
      }
    }
    flushSentences(true); // speak any trailing fragment
    history.push({ role: "assistant", content: full });

    // Wait for speech to drain, then give iOS time to release the audio session
    // before handing the mic back to SpeechRecognition.
    await waitForSpeechEnd();
    await new Promise(r => setTimeout(r, 500));
  } catch (err) {
    reply.textContent = "(Friday had a problem: " + err.message + ")";
    enqueueSpeak("Sorry, I ran into a problem.");
    await waitForSpeechEnd();
    await new Promise(r => setTimeout(r, 500));
  }

  if (active) startListening();
}

function waitForSpeechEnd() {
  return new Promise(resolve => {
    const check = () => {
      if (!isSpeaking()) resolve();
      else setTimeout(check, 150);
    };
    check();
  });
}

// ---------- Voice input (free, on-device STT) ----------
function startListening() {
  if (!active || recognizing) return;
  recognition = buildRecognition();
  startCapture(); // begin accumulating PCM for Eagle identification
  try {
    recognition.start();
  } catch (_) { /* already starting */ }
}

function buildRecognition() {
  const r = new SpeechRecognition();
  r.lang = "en-US";
  r.continuous = false;     // iOS Safari stops after each utterance — we restart on end
  r.interimResults = true;

  r.onstart = () => { recognizing = true; setState("listening", "Listening…"); };

  r.onresult = (event) => {
    let finalText = "";
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) finalText += t;
      else interim += t;
    }
    if (interim) setState("listening", interim);
    if (finalText.trim()) {
      recognition.stop();
      const text = finalText.trim();
      addBubble("user", text);
      // Identify speaker in-browser via Eagle, then chat.
      // Eagle runs in ~50ms so we don't need parallel execution.
      identifySpeaker().then(name => {
        if (name) { currentSpeaker = name; showSpeaker(name); }
        askFriday(text, currentSpeaker);
      });
    }
  };

  r.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      stopSession("Microphone blocked. Allow mic access and reload.");
    }
    // "no-speech" / "aborted" are normal — onend will restart.
  };

  r.onend = () => {
    recognizing = false;
    // Restart only if idle (not thinking/speaking). Delay to debounce rapid loops
    // caused by iOS ending recognition before the audio session is fully released.
    if (active && document.body.className === "listening") {
      setTimeout(startListening, 300);
    }
  };

  return r;
}

// ---------- Session control ----------
function startSession() {
  if (!SpeechRecognition) {
    setState("", "This browser has no speech recognition. Use Safari on iOS or Chrome.");
    return;
  }
  active = true;
  currentSpeaker = null;
  history.length = 0;
  transcriptEl.innerHTML = "";

  // Unlock audio within the tap gesture (iOS requires this).
  unlockAudio();

  // Init mic stream for Eagle PCM capture (Eagle speakers in localStorage → attempt ID).
  const hasSpeakers = loadSpeakerProfiles().length > 0;
  if (hasSpeakers && !captureCtx) {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => initCaptureNode(stream))
      .catch(e => console.warn("Mic for Eagle unavailable:", e.message));
  }

  startListening();
}

function stopSession(message = "Tap to wake Friday") {
  active = false;
  recognizing = false;
  try { recognition && recognition.stop(); } catch (_) {}
  stopSpeaking();
  setState("", message);
}

orb.addEventListener("click", () => {
  if (!active) {
    startSession();
  } else if (isSpeaking()) {
    // Barge-in: stop talking, listen again.
    stopSpeaking();
    startListening();
  } else {
    stopSession();
  }
});
