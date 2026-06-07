// Friday — backend "brain".
//
// A tiny Express server with one job: take the conversation so far, ask Claude,
// and stream the reply back token-by-token (Server-Sent Events) so the browser
// can start *speaking* before the whole answer is done.
//
// The brain (Claude) is the only paid part. Voice in/out happens free, on-device,
// in the browser (see public/app.js).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import Anthropic from "@anthropic-ai/sdk";
import { google } from "googleapis";

// --- minimal .env loader (no extra dependency) ---
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envPath = path.join(__dirname, ".env");
function loadEnv() {
  if (fs.existsSync(envPath)) {
    for (const line of fs.readFileSync(envPath, "utf8").split("\n")) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/i);
      if (m) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
    }
  }
}
loadEnv();

const API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.JARVIS_MODEL || process.env.FRIDAY_MODEL || "claude-opus-4-8";
const PORT = process.env.PORT || 3000;

// Optional: natural voice via ElevenLabs. If no key is set, the browser falls
// back to the free on-device voice automatically.
const ELEVEN_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVEN_VOICE = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; // "Rachel" (a default)
const ELEVEN_MODEL = process.env.ELEVENLABS_MODEL || "eleven_turbo_v2_5"; // low-latency

if (!API_KEY) {
  console.error("\n  ✗ ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.\n");
  process.exit(1);
}

const client = new Anthropic({ apiKey: API_KEY });

// Load the Friday persona once at boot.
const SYSTEM_PROMPT = fs.readFileSync(
  path.join(__dirname, "persona", "system-prompt.md"),
  "utf8"
);

// ─── Google Calendar ──────────────────────────────────────────────────────────

const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const GOOGLE_REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || `http://localhost:${PORT}/auth/google/callback`;

let googleCalendar = null;

function initGoogleCalendar(refreshToken) {
  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET || !refreshToken) return null;
  const auth = new google.auth.OAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI);
  auth.setCredentials({ refresh_token: refreshToken });
  return google.calendar({ version: "v3", auth });
}

googleCalendar = initGoogleCalendar(process.env.GOOGLE_REFRESH_TOKEN);

if (googleCalendar) {
  console.log("  Calendar: Google Calendar connected");
} else if (GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET) {
  console.log(`  Calendar: credentials found — visit http://localhost:${PORT}/auth/google/start to authorize`);
} else {
  console.log("  Calendar: not configured (add GOOGLE_CLIENT_ID + GOOGLE_CLIENT_SECRET to .env)");
}

async function getCalendarEvents(days = 7) {
  const timeMin = new Date().toISOString();
  const timeMax = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toISOString();
  const res = await googleCalendar.events.list({
    calendarId: "primary",
    timeMin,
    timeMax,
    singleEvents: true,
    orderBy: "startTime",
    maxResults: 20,
  });
  return (res.data.items || []).map((e) => ({
    id: e.id,
    summary: e.summary || "(no title)",
    start: e.start?.dateTime || e.start?.date,
    end: e.end?.dateTime || e.end?.date,
    location: e.location || null,
    description: e.description || null,
  }));
}

async function createCalendarEvent({ summary, start, end, description, location }) {
  const res = await googleCalendar.events.insert({
    calendarId: "primary",
    resource: {
      summary,
      location,
      description,
      start: { dateTime: start },
      end: { dateTime: end },
    },
  });
  return { id: res.data.id, summary: res.data.summary, start: res.data.start };
}

// Custom calendar tools exposed to Claude.
const CALENDAR_TOOLS = [
  {
    name: "get_calendar_events",
    description:
      "Get upcoming events from the user's Google Calendar. Use this when the user asks about their schedule, upcoming meetings, or what's on their calendar.",
    input_schema: {
      type: "object",
      properties: {
        days: {
          type: "number",
          description: "How many days ahead to look. Default is 7.",
        },
      },
      required: [],
    },
  },
  {
    name: "create_calendar_event",
    description:
      "Add a new event to the user's Google Calendar. Use this when the user asks to schedule something, book a meeting, or add an event.",
    input_schema: {
      type: "object",
      properties: {
        summary: { type: "string", description: "Event title" },
        start: {
          type: "string",
          description: "Start time in ISO 8601 format, e.g. 2026-06-08T14:00:00+10:00",
        },
        end: {
          type: "string",
          description: "End time in ISO 8601 format",
        },
        description: { type: "string", description: "Optional notes or description" },
        location: { type: "string", description: "Optional location" },
      },
      required: ["summary", "start", "end"],
    },
  },
];

const CALENDAR_TOOL_NAMES = new Set(CALENDAR_TOOLS.map((t) => t.name));

// ─── Express app ──────────────────────────────────────────────────────────────

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// Tell the front-end which voice to use.
app.get("/config", (_req, res) => {
  res.json({ tts: ELEVEN_KEY ? "elevenlabs" : "browser" });
});

// ─── Google OAuth routes (one-time setup) ─────────────────────────────────────

app.get("/auth/google/start", (req, res) => {
  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
    res.status(400).send(
      "<h2>Missing credentials</h2><p>Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to your .env file, then restart Friday.</p>"
    );
    return;
  }
  const oauthClient = new google.auth.OAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI);
  const url = oauthClient.generateAuthUrl({
    access_type: "offline",
    prompt: "consent",
    scope: ["https://www.googleapis.com/auth/calendar"],
  });
  res.redirect(url);
});

app.get("/auth/google/callback", async (req, res) => {
  const { code } = req.query;
  if (!code) {
    res.status(400).send("<h2>No code received.</h2>");
    return;
  }
  try {
    const oauthClient = new google.auth.OAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI);
    const { tokens } = await oauthClient.getToken(String(code));
    if (!tokens.refresh_token) {
      res.status(400).send(
        "<h2>No refresh token returned.</h2><p>You may have already authorized this app. Go to <a href='https://myaccount.google.com/permissions'>Google account permissions</a>, revoke access for this app, then try again.</p>"
      );
      return;
    }
    // Save refresh token to .env
    const current = fs.existsSync(envPath) ? fs.readFileSync(envPath, "utf8") : "";
    const updated = current.includes("GOOGLE_REFRESH_TOKEN=")
      ? current.replace(/^GOOGLE_REFRESH_TOKEN=.*/m, `GOOGLE_REFRESH_TOKEN=${tokens.refresh_token}`)
      : current + `\nGOOGLE_REFRESH_TOKEN=${tokens.refresh_token}\n`;
    fs.writeFileSync(envPath, updated);
    process.env.GOOGLE_REFRESH_TOKEN = tokens.refresh_token;
    oauthClient.setCredentials(tokens);
    googleCalendar = google.calendar({ version: "v3", auth: oauthClient });
    console.log("  Calendar: Google Calendar authorized and ready");
    res.send(
      "<h1>Google Calendar connected!</h1><p>Friday can now read and create calendar events. You can close this tab.</p>"
    );
  } catch (err) {
    console.error("Google OAuth error:", err?.message || err);
    res.status(500).send(`<h2>Auth error</h2><pre>${err?.message}</pre>`);
  }
});

// ─── ElevenLabs TTS proxy ─────────────────────────────────────────────────────

app.post("/tts", async (req, res) => {
  if (!ELEVEN_KEY) {
    res.status(400).json({ error: "ElevenLabs not configured" });
    return;
  }
  const text = (req.body?.text || "").toString().trim();
  if (!text) {
    res.status(400).json({ error: "text required" });
    return;
  }
  try {
    const upstream = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${ELEVEN_VOICE}/stream`,
      {
        method: "POST",
        headers: {
          "xi-api-key": ELEVEN_KEY,
          "Content-Type": "application/json",
          Accept: "audio/mpeg",
        },
        body: JSON.stringify({
          text,
          model_id: ELEVEN_MODEL,
          voice_settings: { stability: 0.4, similarity_boost: 0.8 },
        }),
      }
    );
    if (!upstream.ok || !upstream.body) {
      console.error("ElevenLabs error:", upstream.status, await upstream.text().catch(() => ""));
      res.status(502).json({ error: "tts upstream " + upstream.status });
      return;
    }
    res.setHeader("Content-Type", "audio/mpeg");
    const reader = upstream.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
    res.end();
  } catch (err) {
    console.error("tts error:", err?.message || err);
    if (!res.headersSent) res.status(500).json({ error: "tts failed" });
    else res.end();
  }
});

// ─── Chat endpoint ────────────────────────────────────────────────────────────

// POST /chat  { messages: [{role, content}, ...] }  → SSE stream of {text} chunks
//
// Runs an agentic loop: if Claude wants to use a calendar tool, we execute it
// and loop back. Built-in tools (web_search, web_fetch) are handled transparently
// by Anthropic's infrastructure — no client-side handling needed.
app.post("/chat", async (req, res) => {
  const messages = Array.isArray(req.body?.messages) ? req.body.messages : [];
  if (messages.length === 0) {
    res.status(400).json({ error: "messages array required" });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders?.();

  try {
    const tools = [
      { type: "web_search_20260209", name: "web_search" },
      { type: "web_fetch_20260209", name: "web_fetch" },
      ...(googleCalendar ? CALENDAR_TOOLS : []),
    ];

    let currentMessages = [...messages];
    let iterations = 0;
    const MAX_ITERATIONS = 6;

    while (iterations++ < MAX_ITERATIONS) {
      const response = await client.messages.create({
        model: MODEL,
        max_tokens: 1024,
        system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
        tools,
        messages: currentMessages,
      });

      // Find any calendar tool_use blocks (web_search/web_fetch are handled by Anthropic)
      const calendarCalls = response.content.filter(
        (b) => b.type === "tool_use" && CALENDAR_TOOL_NAMES.has(b.name)
      );

      if (calendarCalls.length === 0) {
        // No custom tools needed — send the final text
        const text = response.content
          .filter((b) => b.type === "text")
          .map((b) => b.text)
          .join("");
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
        res.write("data: [DONE]\n\n");
        return;
      }

      // Execute calendar tools in parallel
      const toolResults = await Promise.all(
        calendarCalls.map(async (block) => {
          try {
            let result;
            if (block.name === "get_calendar_events") {
              const events = await getCalendarEvents(block.input?.days ?? 7);
              result =
                events.length > 0
                  ? JSON.stringify(events)
                  : "No events found in the specified time range.";
            } else if (block.name === "create_calendar_event") {
              const created = await createCalendarEvent(block.input);
              result = JSON.stringify(created);
            }
            return { type: "tool_result", tool_use_id: block.id, content: result };
          } catch (err) {
            return {
              type: "tool_result",
              tool_use_id: block.id,
              content: `Error: ${err.message}`,
              is_error: true,
            };
          }
        })
      );

      currentMessages = [
        ...currentMessages,
        { role: "assistant", content: response.content },
        { role: "user", content: toolResults },
      ];
    }

    // Safety fallback if loop maxes out
    res.write(`data: ${JSON.stringify({ text: "Sorry, I hit an unexpected loop. Try again." })}\n\n`);
    res.write("data: [DONE]\n\n");
  } catch (err) {
    console.error("chat error:", err?.message || err);
    res.write(`data: ${JSON.stringify({ error: err?.message || "model error" })}\n\n`);
  } finally {
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`\n  Friday is listening on http://localhost:${PORT}`);
  console.log(`  Model: ${MODEL}`);
  console.log(`  Voice: ${ELEVEN_KEY ? "ElevenLabs (natural)" : "on-device (browser)"}\n`);
});
