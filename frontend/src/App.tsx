import { useRef, useState } from "react";

function App() {
  const [videoSource, setVideoSource] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleOpenLocalVideo = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const videoURL = URL.createObjectURL(file);
        setVideoSource(videoURL);
      }
    };
    input.click();
  };

  const handleOpenCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Failed to access the camera: ', err);
    }
  };
  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <div className="max-w-3xl p-4 bg-white shadow-lg rounded-lg">
        <h2 className="text-3xl font-bold text-center mb-4">Object Detection</h2>
        <div className = "flex flex-row text-xl justify-between">
        <h2 className="font-bold mb-4">Members:</h2>
        <h2 className="text-base mt-1">21110762 - Cao Thai Dat</h2>
        <h2 className="text-base mt-1">21110792 - Dang Trung Phuong</h2>
        </div>
        <div className="flex flex-row text-xl justify-between">
        <h2 className="ml-20"></h2>
        <h2 className="ml-5 text-base">21110762 - Cao Thai Dat</h2>
        <h2 className="text-base">21110792 - Dang Trung Phuong</h2>
        </div>
        <div className="mb-4">
          <video ref={videoRef} className="w-full h-full object-cover rounded-md" autoPlay />
        </div>
        <div className="flex justify-between">
          <button
            onClick={handleOpenLocalVideo}
            className="px-4 py-2 bg-blue-500 text-white rounded-md"
          >
            Open Local Video
          </button>
          <button
            onClick={handleOpenCamera}
            className="px-4 py-2 bg-green-500 text-white rounded-md"
          >
            Open Camera
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
