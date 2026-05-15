import { useState, useRef } from "react";

export default function App() {
  const [recording, setRecording] = useState(false);
  const [emotion, setEmotion] = useState("");
  const [loading, setLoading] = useState(false);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);


  const startRecording = async () => {
    setEmotion("");
    setLoading(false);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;

    const mediaRecorder = new MediaRecorder(stream);

    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunksRef.current.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      setLoading(true);

      const audioBlob = new Blob(chunksRef.current, {
        type: "audio/webm",
      });

      const formData = new FormData();
      formData.append("file", audioBlob, "voice.webm");

      try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (data.error) {
          setEmotion("Error: " + data.error);
        } else {
          setEmotion(data.emotion);
        }
      } catch (error) {
        setEmotion("Server error");
      }

      setLoading(false);
    };

    mediaRecorder.start();
    setRecording(true);
  };

  // ⛔ STOP RECORDING
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    setRecording(false);
  };

  return (
    <div style={styles.container}>
      <h1> Voice Emotion Recognition</h1>

      <button
        onClick={recording ? stopRecording : startRecording}
        style={styles.button}
      >
        {recording ? "Stop Recording" : "Start Recording"}
      </button>

      {loading && <p>Analyzing voice...</p>}

      {emotion && (
        <div style={styles.result}>
          <h2>Detected Emotion:</h2>
          <h1>{emotion}</h1>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    textAlign: "center",
    marginTop: "100px",
    fontFamily: "Arial",
  },
  button: {
    padding: "15px 30px",
    fontSize: "18px",
    cursor: "pointer",
    marginTop: "20px",
  },
  result: {
    marginTop: "40px",
  },
};