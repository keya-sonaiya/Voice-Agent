import { create } from "zustand";

export type Escalation = {
  escalation?: { reason?: string };
  attempted_answer?: string;
};
type CallState = {
  transcript: string[];
  sentiment: { turn: number; score: number }[];
  escalation: Escalation | null;
  response: string | null;
  addTranscript: (text: string) => void;
  setResponse: (text: string, score?: number) => void;
  setEscalation: (payload: Escalation) => void;
};

export const useCallStore = create<CallState>((set) => ({
  transcript: [],
  sentiment: [],
  escalation: null,
  response: null,
  addTranscript: (text) =>
    set((state) => {
      const line = `Caller: ${text}`;
      return state.transcript.at(-1) === line ? state : { transcript: [...state.transcript, line] };
    }),
  setResponse: (text, score) =>
    set((state) => ({
      response: text,
      transcript: [...state.transcript, `Agent: ${text}`],
      sentiment:
        typeof score === "number"
          ? [...state.sentiment, { turn: state.sentiment.length + 1, score }]
          : state.sentiment,
    })),
  setEscalation: (payload) => set({ escalation: payload }),
}));
