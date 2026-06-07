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

// ── Speaker identification — mel-filterbank fingerprint (no external SDK) ───────
// Records mic audio, computes a 32-band log-mel spectrum per utterance, and
// matches against enrolled templates via cosine similarity. Works entirely in the
// browser; no signup or model file needed.

let currentSpeaker = null;
let captureCtx = null;
let captureProcessor = null;
let captureBuffer = [];
let captureSampleRate = 44100;

const NUM_MEL = 32;
const MEL_F_MIN = 80;
const MEL_F_MAX = 3400;
const FFT_FRAME = 1024;
const ID_THRESHOLD = 0.75;

function hzToMel(hz) { return 1127 * Math.log(1 + hz / 700); }
function melToHz(mel) { return 700 * (Math.exp(mel / 1127) - 1); }

// Iterative Cooley-Tukey FFT in-place on Float32Arrays.
function fftInPlace(re, im) {
  const n = re.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cRe = 1, cIm = 0;
      for (let k = 0; k < (len >> 1); k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + (len >> 1)] * cRe - im[i + k + (len >> 1)] * cIm;
        const vIm = re[i + k + (len >> 1)] * cIm + im[i + k + (len >> 1)] * cRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + (len >> 1)] = uRe - vRe; im[i + k + (len >> 1)] = uIm - vIm;
        const nr = cRe * wRe - cIm * wIm; cIm = cRe * wIm + cIm * wRe; cRe = nr;
      }
    }
  }
}

function buildMelFilters(sampleRate) {
  const numBins = FFT_FRAME / 2;
  const melMin = hzToMel(MEL_F_MIN), melMax = hzToMel(MEL_F_MAX);
  const centers = Array.from({ length: NUM_MEL + 2 }, (_, i) =>
    Math.floor(melToHz(melMin + i * (melMax - melMin) / (NUM_MEL + 1)) * FFT_FRAME / sampleRate)
  );
  return Array.from({ length: NUM_MEL }, (_, m) => {
    const f = new Float32Array(numBins);
    for (let k = centers[m]; k < centers[m + 1]; k++) f[k] = (k - centers[m]) / Math.max(1, centers[m + 1] - centers[m]);
    for (let k = centers[m + 1]; k < centers[m + 2]; k++) f[k] = (centers[m + 2] - k) / Math.max(1, centers[m + 2] - centers[m + 1]);
    return f;
  });
}

function computeTemplate(float32, sampleRate) {
  if (!float32 || float32.length < FFT_FRAME * 2) return null;
  const filters = buildMelFilters(sampleRate);
  const numBins = FFT_FRAME / 2;
  const hop = FFT_FRAME >> 2;
  const avg = new Float32Array(NUM_MEL);
  let n = 0;
  for (let start = 0; start + FFT_FRAME <= float32.length; start += hop) {
    const re = float32.slice(start, start + FFT_FRAME);
    for (let i = 0; i < FFT_FRAME; i++) re[i] *= 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_FRAME);
    const im = new Float32Array(FFT_FRAME);
    fftInPlace(re, im);
    for (let m = 0; m < NUM_MEL; m++) {
      let e = 0;
      for (let k = 0; k < numBins; k++) e += filters[m][k] * Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      avg[m] += Math.log(e + 1e-8);
    }
    n++;
  }
  if (!n) return null;
  for (let m = 0; m < NUM_MEL; m++) avg[m] /= n;
  let norm = 0;
  for (let m = 0; m < NUM_MEL; m++) norm += avg[m] * avg[m];
  norm = Math.sqrt(norm) || 1;
  for (let m = 0; m < NUM_MEL; m++) avg[m] /= norm;
  return avg;
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

function templateToB64(t) {
  const bytes = new Uint8Array(t.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s);
}

function b64ToTemplate(b64) {
  const s = atob(b64);
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function loadSpeakerProfiles() {
  try { return JSON.parse(localStorage.getItem("friday_voice_profiles") || "[]"); } catch (_) { return []; }
}

function startCapture() {
  captureBuffer = [];
  if (!captureCtx || !captureProcessor) return;
  captureProcessor.onaudioprocess = (e) => {
    captureBuffer.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
}

function stopCapture() {
  if (captureProcessor) captureProcessor.onaudioprocess = null;
  if (!captureBuffer.length) return null;
  const total = captureBuffer.reduce((s, c) => s + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of captureBuffer) { out.set(c, offset); offset += c.length; }
  captureBuffer = [];
  return out;
}

async function identifySpeaker() {
  const profiles = loadSpeakerProfiles();
  console.log(`[SpeakerID] called — profiles=${profiles.length} captureCtx=${!!captureCtx}`);
  if (!profiles.length) { console.log("[SpeakerID] no profiles enrolled"); return null; }
  const audio = stopCapture();
  if (!audio) { console.log("[SpeakerID] no audio captured"); return null; }
  console.log(`[SpeakerID] audio=${audio.length} samples @ ${captureSampleRate}Hz`);
  const tmpl = computeTemplate(audio, captureSampleRate);
  if (!tmpl) { console.log("[SpeakerID] template computation failed"); return null; }
  let best = null, bestScore = -Infinity;
  for (const p of profiles) {
    const score = cosineSim(tmpl, b64ToTemplate(p.templateB64));
    if (score > bestScore) { bestScore = score; best = p.name; }
  }
  console.log(`[SpeakerID] best="${best}" score=${bestScore.toFixed(3)} threshold=${ID_THRESHOLD}`);
  return bestScore >= ID_THRESHOLD ? best : null;
}

async function initCaptureNode(stream) {
  try {
    captureCtx = new (window.AudioContext || window.webkitAudioContext)();
    captureSampleRate = captureCtx.sampleRate;
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

  // Init mic stream for voice fingerprint capture.
  const hasSpeakers = loadSpeakerProfiles().length > 0;
  console.log(`[SpeakerID] session start — hasSpeakers=${hasSpeakers}`, loadSpeakerProfiles().map(p => p.name));
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
