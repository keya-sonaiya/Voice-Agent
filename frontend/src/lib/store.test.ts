import { describe, expect, it } from "vitest";
import { useCallStore } from "./store";

describe("call store", () => {
  it("shares response and sentiment state", () => {
    useCallStore.getState().setResponse("A grounded answer", -0.2);
    expect(useCallStore.getState().sentiment.at(-1)).toEqual({
      turn: 1,
      score: -0.2,
    });
  });
});
