const API_URL = import.meta.env.VITE_API_BASE;

export async function getPredictionHistory() {
  const token = localStorage.getItem("accessToken");
  const API_URL = import.meta.env.VITE_API_BASE;

  const res = await fetch(`${API_URL}/predict/history`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!res.ok) {
    throw new Error("Gagal mengambil prediction history");
  }

  const data = await res.json();
  return data.data || [];
}
