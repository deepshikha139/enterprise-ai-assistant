"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// One complete Server-Sent Event, e.g. "event: delta\ndata: {...}"
function parseEvent(
  raw: string,
): { event: string; data: { text?: string; message?: string } } | null {
  let event = "";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    if (line.startsWith("data: ")) data = line.slice(6);
  }
  if (!event || !data) return null;
  return { event, data: JSON.parse(data) };
}

export default function Home() {
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function send() {
    setAnswer("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      // 503 / 500 before the stream starts: a normal JSON error.
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        setError(body?.detail ?? "Something went wrong.");
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // stream: true keeps a half-received character for the next read.
        buffer += decoder.decode(value, { stream: true });

        // Events end with a blank line. The last part may be unfinished,
        // so it stays in the buffer until the next read completes it.
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          const parsed = parseEvent(part);
          if (!parsed) continue;

          if (parsed.event === "delta") {
            // Use the previous value: this loop runs outside React's render.
            setAnswer((prev) => prev + (parsed.data.text ?? ""));
          } else if (parsed.event === "done") {
            finished = true;
          } else if (parsed.event === "error") {
            finished = true;
            setError(parsed.data.message ?? "Something went wrong.");
          }
        }
      }

      if (!finished) setError("The answer was cut off.");
    } catch {
      setError("Could not reach the server.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Ask a question"
      />
      <button onClick={send} disabled={loading || !message.trim()}>
        Send
      </button>
      <p style={{ whiteSpace: "pre-wrap" }}>{answer}</p>
      {error && <p role="alert">{error}</p>}
    </main>
  );
}
