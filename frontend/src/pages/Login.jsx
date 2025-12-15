import { useState } from "react";
import { useNavigate } from "react-router-dom";
import LoginForm from "../components/LoginForm";
import AnimatedBackground from "../components/AnimatedBackground";

const API_BASE = import.meta.env.VITE_API_BASE;

export default function Login() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const togglePasswordVisibility = () =>
    setShowPassword((prev) => !prev);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const json = await res.json();

      if (json.error) {
        setError(json.error || "Login gagal. Coba lagi.");
        return;
      }

      localStorage.setItem("accessToken", json.accessToken);
      navigate("/dashboard");
    } catch (err) {
      setError("Gagal terhubung ke server.");
      console.error("Login error:", err);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#0a0f18] text-white overflow-y-auto px-4">
      <AnimatedBackground />

      <LoginForm
        form={form}
        handleChange={handleChange}
        handleSubmit={handleSubmit}
        error={error}
        showPassword={showPassword}
        togglePasswordVisibility={togglePasswordVisibility}
      />
    </div>
  );
}
