import { useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const [message, setMessage] = useState("");

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/api/hello")
      .then(res => setMessage(res.data.response))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <h1 className="text-2xl font-bold">{message || "Connecting..."}</h1>
    </div>
  );
}