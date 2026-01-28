import React, {useRef, useEffect, useState} from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const photoRef = useRef(null);

  const [hasPhoto, setHasPhoto] = useState(false);

  const getVideo = () => {
    navigator.mediaDevices
    .getUserMedia({
      video: {width: 1920, height: 1080}
    })
    .then(stream => {
      let video = videoRef.current;
      video.srcObject = stream;
      video.play();
    })
    .catch(err => {
      console.error(err);
    })
  }

  const takePhoto = () => {
    const width = 414;
    const height = width / (16/9);

    let video = videoRef.current;
    let photo = photoRef.current;

    photo.width = width;
    photo.height = height;

    let ctx = photo.getContext('2d');
    ctx.drawImage(video, 0, 0, width, height);
    setHasPhoto(true);
  }

  const closePhoto = () => {
    let photo = photoRef.current;
    let ctx = photo.getContext('2d');

    ctx.clearRect(0, 0, photo.width, photo.height);

    setHasPhoto(false);
  }

  useEffect(() => {
    getVideo();
  }, [videoRef]);

  return (
    <div style={{position: 'relative'}}>

      <div className="App">
        
        <header>
          <title>OCR Calculator</title>
        </header>

        <div className="title">
          <h1>OCR Calculator</h1>
        </div>

        <div className="camera">
          <video ref={videoRef}></video>
            <div className="photo-button" onClick={takePhoto}>
              <div className="circle"></div>
              <div className="ring"></div>
            </div>
        </div>
        
        <div className={'result' + (hasPhoto ? 'hasPhoto' : '')}>
          <canvas ref={photoRef}></canvas>
          <button onClick={closePhoto}>Close</button>
        </div>

        <div className="expressionDetected">
          <h1>Expression Detected:</h1>
          <h2>temp</h2>
        </div>

        <div className="solution">
          <h1>Solution:</h1>   
          <h2>temp</h2>       
        </div>


      </div>

    </div>

  );
}

export default App;
