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
  setResponse: (text: string, score: number) => void;
  setEscalation: (payload: Escalation) => void;
};

export const useCallStore = create<CallState>((set) => ({
  transcript: [],
  sentiment: [],
  escalation: null,
  response: null,
  addTranscript: (text) =>
    set((state) => ({ transcript: [...state.transcript, `Caller: ${text}`] })),
  setResponse: (text, score) =>
    set((state) => ({
      response: text,
      transcript: [...state.transcript, `Agent: ${text}`],
      sentiment: [
        ...state.sentiment,
        { turn: state.sentiment.length + 1, score },
      ],
    })),
  setEscalation: (payload) => set({ escalation: payload }),
}));
