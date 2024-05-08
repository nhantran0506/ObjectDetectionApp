import { useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";

function App() {
  //   const webcamRef = useRef<Webcam>(null);
  //   const [videoSource, setVideoSource] = useState<string | null>(null);
  //   const [isCameraEnabled, setCameraEnabled] = useState<boolean>(false);
  const [url, setUrl] = useState<string | null>(null);
  const members: { id: number; name: string; leader: boolean }[] = [
    { id: 21110790, name: "Trần Trọng Nhân", leader: true },
    { id: 21110809, name: "Nguyễn Phước Trường", leader: false },
    { id: 21110762, name: "Cao Thái Đạt", leader: false },
    { id: 21110792, name: "Đặng Trung Phương", leader: false },
  ];
  const memberElements = members.map(function (member) {
    return (
      <tr key={member.id} className="border-b bg-white ">
        <td className=" whitespace-nowrap px-6 py-3 text-gray-900" scope="row">
          {member.name}
        </td>
        <td className=" px-6 py-3" scope="row">
          {member.id}
        </td>
        <td className=" px-6 py-3 font-black" scope="row">
          {member.leader ? "*" : ""}
        </td>
      </tr>
    );
  });

  //   const handleVideoUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
  //     const file = event.target.files?.[0];
  //     setVideoSource(file ? URL.createObjectURL(file) : null);
  //   };

  return (
    <BrowserRouter>
      <div className="flex h-screen w-screen items-center justify-center bg-gray-50 p-10">
        <div className="grid rounded-lg bg-white p-4 shadow-md">
          <h1 className="m-3 text-center text-5xl font-bold">
            <a href="/">Object Detection Application</a>
          </h1>
          <Routes>
            <Route
              path="/*"
              element={
                <div className="mx-10 mb-3 grid justify-center">
                  <h2 className="mb-3 text-center text-3xl font-bold">Group 10</h2>
                  <table className=" w-full table-auto place-self-center text-left text-xl text-gray-500">
                    <thead className="bg-gray-50 text-gray-700">
                      <tr>
                        <th className=" px-6 py-2" scope="col">
                          Name
                        </th>
                        <th className=" px-6 py-2" scope="col">
                          ID
                        </th>
                        <th className=" px-6 py-2" scope="col">
                          Leader
                        </th>
                      </tr>
                    </thead>
                    <tbody>{memberElements}</tbody>
                  </table>
                </div>
              }
            ></Route>
            {/* <Route path="/" element={<div className="grid"></div>}></Route> */}
          </Routes>
          {/* <input
            type="file"
            accept="video/*"
            onChange={handleVideoUpload}
            style={{ display: "none" }}
            id="video-upload"
          /> */}
          {/* <div className="hidden grid-cols-2 gap-4">
            <button
              className="rounded-md bg-gray-200 p-2 transition duration-200 hover:bg-gray-400"
              onClick={() => {
                setCameraEnabled(false);
                document.getElementById("video-upload")?.click();
              }}
            >
              Open Video File
            </button>
            <button
              className="rounded-md bg-gray-200 p-2 transition duration-200 hover:bg-gray-400"
              onClick={() => {
                setVideoSource(null);
                setCameraEnabled(true);
              }}
            >
              Open Camera
            </button>
          </div> */}
          <div className="m-3 h-max grid justify-center">
            <form
              action="/upload"
              method="post"
              encType="multipart/form-data"
              className="w-fit mx-10 flex justify-center"
            >
              <input
                className="block h-auto w-full cursor-pointer rounded-lg border border-gray-300 bg-gray-50 p-2 text-sm text-gray-900 focus:outline-none"
                id="file_input"
                type="file"
                name="video"
                accept="video/mp4"
              />

              {/* <input type="file" name="video" accept="video/mp4" className="block text-md" /> */}
              <input
                type="submit"
                value="Upload"
                className="hover: rounded-lg border p-2 shadow transition duration-100 ease-in-out hover:bg-black hover:text-white"
              />
            </form>
          </div>
          <Routes>
            <Route path="/" element={<div className="flex"></div>}></Route>
            <Route
              path="/video"
              element={
                <div>
                  <div className="m-3 flex h-auto w-fit justify-center">
                    <img src="/video_feed" className="h-auto w-max rounded-lg"></img>
                  </div>
                </div>
              }
            ></Route>

            <Route path="/*" element={<div className="grid"></div>}></Route>
          </Routes>
          {/* <div className="flex h-auto w-[720px] justify-center">
            <img src="/video_feed" className="h-auto w-max rounded-lg"></img>
          </div> */}
        </div>
        {url && (
          <>
            <div>
              <button
                onClick={() => {
                  setUrl(null);
                }}
              >
                delete
              </button>
            </div>
            <div>
              <img src={url} alt="Screenshot" />
            </div>
          </>
        )}
      </div>
    </BrowserRouter>
  );
}

export default App;
