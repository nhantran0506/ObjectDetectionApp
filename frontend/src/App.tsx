import { useState } from "react";

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="flex h-screen flex-col items-center justify-center">
      <h1 className=" text-3xl font-bold">Object Detection App</h1>
      <div className="card">
        <button
          className="rounded bg-purple-500 px-4 py-2 font-bold text-white hover:bg-purple-700"
          onClick={() => setCount((count) => count + 1)}
        >
          count is {count}
        </button>
      </div>
    </div>
  );
}

export default App;
