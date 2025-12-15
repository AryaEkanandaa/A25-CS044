const API_BASE = import.meta.env.VITE_API_BASE;

export async function askChatbot(message) {
  const res = await fetch(`${API_BASE}/chatbot`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  return await res.json();
}
