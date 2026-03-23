const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const jsonOutput = document.getElementById('jsonOutput');

canvas.width = 640;
canvas.height = 480;

let frameIndex = 0;

const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

// 🔥 Send to backend
async function sendFrame(data) {
  try {
    await fetch('http://localhost:3000/hand-data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
  } catch (err) {
    console.log('Error sending:', err);
  }
}

hands.onResults(results => {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      drawConnectors(ctx, results.multiHandLandmarks[i], HAND_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 2
      });
      drawLandmarks(ctx, results.multiHandLandmarks[i], {
        color: '#FF0000',
        lineWidth: 0.5,
        radius: 2
      });
    }
  }

  ctx.restore();

  const keypointFrame = {
    sequence_id: "uuid-123",
    timestamp: Date.now(),
    frame_index: frameIndex,
    left_hand_landmarks: results.multiHandLandmarks?.[0] || [],
    right_hand_landmarks: results.multiHandLandmarks?.[1] || []
  };

  // ✅ Keep original behavior
  jsonOutput.textContent = JSON.stringify(keypointFrame, null, 2);

  // ✅ New feature
  sendFrame(keypointFrame);

  frameIndex++;
});

const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({ image: video });
  },
  width: 640,
  height: 480
});

camera.start();