"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { AlertTriangle, Mic, Phone, Send } from "lucide-react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { z } from "zod";

import { useCallStore } from "../lib/store";

const eventSchema = z.object({
  type: z.string(),
  text: z.string().optional(),
  final: z.boolean().optional(),
  sentiment: z.number().optional(),
  payload: z
    .object({
      escalation: z.object({ reason: z.string().optional() }).optional(),
      attempted_answer: z.string().optional(),
    })
    .optional(),
});

export default function Home() {
  const socket = useRef<WebSocket | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const microphoneStream = useRef<MediaStream | null>(null);
  const microphoneNode = useRef<ScriptProcessorNode | null>(null);
  const microphoneMuteNode = useRef<GainNode | null>(null);
  const [draft, setDraft] = useState("");
  const [connected, setConnected] = useState(false);
  const [starting, setStarting] = useState(false);
  const [microphoneActive, setMicrophoneActive] = useState(false);
  const {
    transcript,
    sentiment,
    escalation,
    addTranscript,
    setResponse,
    setEscalation,
  } = useCallStore();

  useEffect(() => {
    return () => {
      microphoneNode.current?.disconnect();
      microphoneMuteNode.current?.disconnect();
      microphoneStream.current?.getTracks().forEach((track) => track.stop());
      void audioContext.current?.close();
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
      const sample =
        samples[lower] * (1 - fraction) + samples[upper] * fraction;
      output[index] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
    }
    return output.buffer;
  }

  async function startMicrophone() {
    if (socket.current?.readyState !== WebSocket.OPEN || microphoneActive)
      return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(2048, 1, 1);
    const muteNode = context.createGain();
    muteNode.gain.value = 0;
    processor.onaudioprocess = (event) => {
      if (socket.current?.readyState === WebSocket.OPEN) {
        socket.current.send(
          pcm16k(event.inputBuffer.getChannelData(0), context.sampleRate),
        );
      }
    };
    source.connect(processor);
    // Keep the processor alive without playing the caller's microphone back
    // through their speakers, which can degrade recognition through feedback.
    processor.connect(muteNode);
    muteNode.connect(context.destination);
    microphoneStream.current = stream;
    microphoneNode.current = processor;
    microphoneMuteNode.current = muteNode;
    audioContext.current = context;
    setMicrophoneActive(true);
  }

  function stopMicrophone() {
    microphoneNode.current?.disconnect();
    microphoneMuteNode.current?.disconnect();
    microphoneStream.current?.getTracks().forEach((track) => track.stop());
    void audioContext.current?.close();
    microphoneNode.current = null;
    microphoneMuteNode.current = null;
    microphoneStream.current = null;
    audioContext.current = null;
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: "audio_end" }));
    }
    setMicrophoneActive(false);
  }

  async function startCall() {
    setStarting(true);
    try {
      const api =
        process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const result = await fetch(`${api}/calls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caller_id: "demo-caller" }),
      });
      if (!result.ok) throw new Error("Could not start a call.");
      const { session_id, token } = (await result.json()) as {
        session_id: string;
        token: string;
      };
      const base =
        process.env.NEXT_PUBLIC_WS_GATEWAY_URL ??
        "ws://localhost:8000/ws/audio";
      const ws = new WebSocket(`${base}/${session_id}`);
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "auth", token }));
        setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        setMicrophoneActive(false);
      };
      ws.onmessage = (message) => {
        if (typeof message.data !== "string") return;
        let data: unknown;
        try {
          data = JSON.parse(message.data);
        } catch {
          return;
        }
        const parsed = eventSchema.safeParse(data);
        if (!parsed.success) return;
        const event = parsed.data;
        if (event.type === "transcript" && event.text && event.final)
          addTranscript(event.text);
        if (
          event.type === "response" &&
          event.text &&
          event.sentiment !== undefined
        )
          setResponse(event.text, event.sentiment);
        if (event.type === "escalation" && event.payload)
          setEscalation(event.payload);
      };
      socket.current = ws;
    } finally {
      setStarting(false);
    }
  }

  function sendTranscript(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim() || socket.current?.readyState !== WebSocket.OPEN) return;
    socket.current.send(JSON.stringify({ type: "transcript", text: draft }));
    setDraft("");
  }

  return (
    <main>
      <section className="hero">
        <span>VOICE · SUPPORT</span>
        <h1>Support that listens, verifies, and knows when to hand off.</h1>
        <p>
          A live LangGraph pipeline with grounding and human escalation gates.
        </p>
      </section>
      <section className="panel controls">
        <div>
          <strong>{connected ? "Call connected" : "Ready to start"}</strong>
          <p>
            Secure WebSocket session ·{" "}
            {connected ? "listening" : "not connected"}
          </p>
        </div>
        <div className="actions">
          <button onClick={startCall} disabled={connected || starting}>
            <Phone size={18} /> {starting ? "Connecting…" : "Start demo call"}
          </button>
          {connected && (
            <button
              className="secondary"
              onClick={
                microphoneActive ? stopMicrophone : () => void startMicrophone()
              }
            >
              <Mic size={18} />{" "}
              {microphoneActive ? "Stop microphone" : "Use microphone"}
            </button>
          )}
        </div>
      </section>
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
            {transcript.length ? (
              transcript.map((line, index) => (
                <p key={`${line}-${index}`}>{line}</p>
              ))
            ) : (
              <p className="muted">
                Start the demo, then speak or type a support request.
              </p>
            )}
          </div>
          <form onSubmit={sendTranscript}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Type a transcript for the demo"
              disabled={!connected}
            />
            <button aria-label="Send transcript" disabled={!connected}>
              <Send size={18} />
            </button>
          </form>
          <p className="mic">
            <Mic size={15} /> Microphone audio is converted to 16 kHz PCM and
            sent on the call WebSocket.
          </p>
        </div>
        <div className="panel">
          <h2>Caller sentiment</h2>
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sentiment}>
                <XAxis dataKey="turn" />
                <YAxis domain={[-1, 1]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke="#30d5a5"
                  strokeWidth={3}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </main>
  );
}
