"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertTriangle, Mic, Phone, Send } from "lucide-react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { z } from "zod";

import { useCallStore } from "../lib/store";

const eventSchema = z
  .object({
    type: z.string(),
    text: z.string().optional(),
    final: z.boolean().optional(),
    sentiment: z.number().optional(),
    stage: z.string().optional(),
    message: z.string().optional(),
    format: z.string().optional(),
    payload: z
      .object({
        escalation: z.object({ reason: z.string().optional() }).optional(),
        attempted_answer: z.string().optional(),
      })
      .optional(),
  })
  .passthrough();

const debugLogging =
  process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true" || process.env.NODE_ENV !== "production";

function wsLog(event: string, details?: Record<string, unknown>) {
  if (debugLogging) console.debug(`[WS] ${event}`, details ?? {});
}

function wsWarn(event: string, details?: Record<string, unknown>) {
  if (debugLogging) console.warn(`[WS] ${event}`, details ?? {});
}

export default function Home() {
  const socket = useRef<WebSocket | null>(null);
  const microphoneAudioContext = useRef<AudioContext | null>(null);
  const playbackAudioContext = useRef<AudioContext | null>(null);
  const microphoneStream = useRef<MediaStream | null>(null);
  const microphoneNode = useRef<ScriptProcessorNode | null>(null);
  const microphoneMuteNode = useRef<GainNode | null>(null);
  const playbackChunks = useRef<ArrayBuffer[]>([]);
  const playbackSource = useRef<AudioBufferSourceNode | null>(null);
  const playbackGeneration = useRef(0);
  const microphoneFrameCount = useRef(0);
  const microphoneByteCount = useRef(0);
  const [draft, setDraft] = useState("");
  const [connected, setConnected] = useState(false);
  const [starting, setStarting] = useState(false);
  const [microphoneActive, setMicrophoneActive] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const { transcript, sentiment, escalation, addTranscript, setResponse, setEscalation } = useCallStore();

  useEffect(() => {
    return () => {
      microphoneNode.current?.disconnect();
      microphoneMuteNode.current?.disconnect();
      microphoneStream.current?.getTracks().forEach((track) => track.stop());
      playbackSource.current?.stop();
      void microphoneAudioContext.current?.close();
      void playbackAudioContext.current?.close();
      socket.current?.close();
    };
  }, []);

  function pcm16k(samples: Float32Array, sourceRate: number): ArrayBuffer {
    const targetLength = Math.round((samples.length * 16_000) / sourceRate);
    const output = new Int16Array(targetLength);
    for (let index = 0; index < targetLength; index += 1) {
      const position = (index * sourceRate) / 16_000;
      const lower = Math.floor(position);
      const upper = Math.min(lower + 1, samples.length - 1);
      const fraction = position - lower;
      const sample = samples[lower] * (1 - fraction) + samples[upper] * fraction;
      output[index] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
    }
    return output.buffer;
  }

  function stopPlayback() {
    playbackGeneration.current += 1;
    playbackChunks.current = [];
    playbackSource.current?.stop();
    playbackSource.current = null;
  }

  async function playBufferedTts() {
    const chunks = playbackChunks.current;
    playbackChunks.current = [];
    if (!chunks.length) {
      wsWarn("tts_complete without audio chunks");
      return;
    }
    const generation = playbackGeneration.current;
    const totalLength = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
    const audio = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      audio.set(new Uint8Array(chunk), offset);
      offset += chunk.byteLength;
    }
    try {
      const context = playbackAudioContext.current ?? new AudioContext();
      playbackAudioContext.current = context;
      await context.resume();
      wsLog("audio decode started", { bytes: totalLength });
      const decoded = await context.decodeAudioData(audio.buffer);
      wsLog("audio decoded", { durationSeconds: decoded.duration });
      if (generation !== playbackGeneration.current) return;
      const source = context.createBufferSource();
      source.buffer = decoded;
      source.connect(context.destination);
      source.onended = () => {
        if (playbackSource.current === source) playbackSource.current = null;
        wsLog("playback ended");
      };
      playbackSource.current = source;
      source.start();
      wsLog("playback started");
    } catch (error) {
      wsWarn("audio decode or playback failed", { reason: String(error) });
      setNotice("The text response is available, but speech playback could not be started.");
    }
  }

  async function startMicrophone() {
    if (socket.current?.readyState !== WebSocket.OPEN || microphoneActive) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(2048, 1, 1);
      const muteNode = context.createGain();
      muteNode.gain.value = 0;
      microphoneFrameCount.current = 0;
      microphoneByteCount.current = 0;
      processor.onaudioprocess = (event) => {
        if (socket.current?.readyState !== WebSocket.OPEN) return;
        const pcm = pcm16k(event.inputBuffer.getChannelData(0), context.sampleRate);
        microphoneFrameCount.current += 1;
        microphoneByteCount.current += pcm.byteLength;
        socket.current.send(pcm);
        wsLog("microphone PCM frame sent", {
          frameCount: microphoneFrameCount.current,
          chunkBytes: pcm.byteLength,
          accumulatedBytes: microphoneByteCount.current,
        });
      };
      source.connect(processor);
      // Keep ScriptProcessor active while keeping microphone input out of the speakers.
      processor.connect(muteNode);
      muteNode.connect(context.destination);
      microphoneStream.current = stream;
      microphoneNode.current = processor;
      microphoneMuteNode.current = muteNode;
      microphoneAudioContext.current = context;
      setMicrophoneActive(true);
      wsLog("microphone started", { sampleRate: context.sampleRate });
    } catch (error) {
      wsWarn("microphone start failed", { reason: String(error) });
      setNotice("Microphone access failed. You can still send a typed transcript.");
    }
  }

  function stopMicrophone() {
    microphoneNode.current?.disconnect();
    microphoneMuteNode.current?.disconnect();
    microphoneStream.current?.getTracks().forEach((track) => track.stop());
    void microphoneAudioContext.current?.close();
    microphoneNode.current = null;
    microphoneMuteNode.current = null;
    microphoneStream.current = null;
    microphoneAudioContext.current = null;
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: "audio_end" }));
      wsLog("audio_end sent", {
        frameCount: microphoneFrameCount.current,
        accumulatedBytes: microphoneByteCount.current,
      });
    }
    setMicrophoneActive(false);
  }

  async function handleSocketMessage(message: MessageEvent) {
    if (message.data instanceof Blob) {
      const chunk = await message.data.arrayBuffer();
      playbackChunks.current.push(chunk);
      wsLog("TTS binary chunk received", { bytes: chunk.byteLength, chunks: playbackChunks.current.length });
      return;
    }
    if (message.data instanceof ArrayBuffer) {
      playbackChunks.current.push(message.data);
      wsLog("TTS binary chunk received", { bytes: message.data.byteLength, chunks: playbackChunks.current.length });
      return;
    }
    if (typeof message.data !== "string") {
      wsWarn("unsupported WebSocket payload", { dataType: typeof message.data });
      return;
    }
    wsLog("JSON message received", { length: message.data.length });
    let data: unknown;
    try {
      data = JSON.parse(message.data);
    } catch (error) {
      wsWarn("malformed WebSocket JSON rejected", { reason: String(error), payload: message.data.slice(0, 200) });
      return;
    }
    const parsed = eventSchema.safeParse(data);
    if (!parsed.success) {
      wsWarn("WebSocket event schema rejected", { reason: parsed.error.issues.map((issue) => issue.message).join(", ") });
      return;
    }
    const event = parsed.data;
    wsLog("event parsed", { type: event.type });
    if (event.type === "transcript" && event.text && event.final) {
      addTranscript(event.text);
      wsLog("transcript received", { textLength: event.text.length });
    }
    if (event.type === "response" && event.text) {
      stopPlayback();
      setResponse(event.text, event.sentiment);
      setNotice(null);
      wsLog("response received", { textLength: event.text.length, hasSentiment: event.sentiment !== undefined });
    }
    if (event.type === "tts_started") wsLog("TTS started", { format: event.format });
    if (event.type === "tts_complete") {
      wsLog("tts_complete received", { chunks: playbackChunks.current.length });
      await playBufferedTts();
    }
    if (event.type === "tts_interrupted") {
      stopPlayback();
      wsLog("TTS interrupted");
    }
    if (event.type === "escalation" && event.payload) {
      setEscalation(event.payload);
      wsLog("escalation received");
    }
    if (event.type === "backend_error") {
      setNotice(event.message ?? "The response pipeline failed. Check backend logs.");
      wsWarn("backend_error received", { stage: event.stage });
    }
    if (event.type === "tts_unavailable") {
      setNotice(event.message ?? "Speech playback is temporarily unavailable.");
      wsWarn("tts_unavailable received");
    }
  }

  async function startCall() {
    setStarting(true);
    setNotice(null);
    try {
      const api = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const result = await fetch(`${api}/calls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caller_id: "demo-caller" }),
      });
      if (!result.ok) throw new Error("Could not start a call.");
      const { session_id, token } = (await result.json()) as { session_id: string; token: string };
      const base = process.env.NEXT_PUBLIC_WS_GATEWAY_URL ?? "ws://localhost:8000/ws/audio";
      const ws = new WebSocket(`${base}/${session_id}`);
      ws.onopen = () => {
        wsLog("connected", { sessionId: session_id });
        ws.send(JSON.stringify({ type: "auth", token }));
        wsLog("authentication sent");
        setConnected(true);
      };
      ws.onclose = () => {
        wsLog("disconnected");
        setConnected(false);
        setMicrophoneActive(false);
      };
      ws.onerror = () => {
        wsWarn("WebSocket error");
        setNotice("The call connection encountered an error. Check backend logs.");
      };
      ws.onmessage = (message) => {
        void handleSocketMessage(message).catch((error) => {
          wsWarn("message handler failed", { reason: String(error) });
          setNotice("A response message could not be processed.");
        });
      };
      socket.current = ws;
    } catch (error) {
      wsWarn("call start failed", { reason: String(error) });
      setNotice("Could not start the call. Check that the backend is running.");
    } finally {
      setStarting(false);
    }
  }

  function sendTranscript(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim() || socket.current?.readyState !== WebSocket.OPEN) return;
    socket.current.send(JSON.stringify({ type: "transcript", text: draft }));
    wsLog("typed transcript sent", { textLength: draft.trim().length });
    setDraft("");
  }

  return (
    <main>
      <section className="hero">
        <span>VOICE · SUPPORT</span>
        <h1>Support that listens, verifies, and knows when to hand off.</h1>
        <p>A live LangGraph pipeline with grounding and human escalation gates.</p>
      </section>
      <section className="panel controls">
        <div>
          <strong>{connected ? "Call connected" : "Ready to start"}</strong>
          <p>Secure WebSocket session · {connected ? "listening" : "not connected"}</p>
        </div>
        <div className="actions">
          <button onClick={() => void startCall()} disabled={connected || starting}>
            <Phone size={18} /> {starting ? "Connecting…" : "Start demo call"}
          </button>
          {connected && (
            <button className="secondary" onClick={microphoneActive ? stopMicrophone : () => void startMicrophone()}>
              <Mic size={18} /> {microphoneActive ? "Stop microphone" : "Use microphone"}
            </button>
          )}
        </div>
      </section>
      {notice && <p className="notice">{notice}</p>}
      {escalation && (
        <section className="escalation">
          <AlertTriangle size={20} />
          <div>
            <strong>Human handoff requested</strong>
            <p>Reason: {escalation.escalation?.reason ?? "support review"}</p>
          </div>
        </section>
      )}
      <section className="grid">
        <div className="panel">
          <h2>Live conversation</h2>
          <div className="transcript">
            {transcript.length ? transcript.map((line, index) => <p key={`${line}-${index}`}>{line}</p>) : <p className="muted">Start the demo, then speak or type a support request.</p>}
          </div>
          <form onSubmit={sendTranscript}>
            <input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Type a transcript for the demo" disabled={!connected} />
            <button aria-label="Send transcript" disabled={!connected}><Send size={18} /></button>
          </form>
          <p className="mic"><Mic size={15} /> Microphone audio is converted to 16 kHz PCM and sent on the call WebSocket.</p>
        </div>
        <div className="panel">
          <h2>Caller sentiment</h2>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sentiment}>
                <XAxis dataKey="turn" /><YAxis domain={[-1, 1]} /><Tooltip />
                <Line type="monotone" dataKey="score" stroke="#30d5a5" strokeWidth={3} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </main>
  );
}
