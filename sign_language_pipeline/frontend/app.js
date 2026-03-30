// ---------- Get HTML elements (UI + video/canvas) ----------km 
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// UI elements for displaying status & results
const jsonOutput = document.getElementById('jsonOutput');
const statusText = document.getElementById('status');
const frameCountText = document.getElementById('frameCount');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');

// Control buttons
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

// Set canvas size (must match video for proper overlay)
canvas.width = 640;
canvas.height = 480;

// ---------- Core configuration ----------
const NUM_FRAMES = 30; // Number of frames required by LSTM model

let frameIndex = 0; // Tracks current frame number
let sequenceBuffer = []; // Stores sequence of frames (each = 225 features)
let isPredicting = false; // Whether user has started prediction
let isSending = false; // Prevent duplicate API calls

// ---------- Initial UI ----------
statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

// ---------- Initialize MediaPipe Holistic ----------
const holistic = new Holistic({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

// Configure detection/tracking settings
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// ---------- Button: Start Prediction ----------
startBtn.addEventListener('click', () => {
  isPredicting = true;
  isSending = false;
  sequenceBuffer = [];
  frameIndex = 0;

  frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
  predictionText.textContent = 'Collecting frames...';
  confidenceText.textContent = 'Confidence: --';
  statusText.textContent = 'Prediction started. Please perform a sign.';
});

// ---------- Button: Stop Prediction ----------
stopBtn.addEventListener('click', () => {
  isPredicting = false;
  isSending = false;
  sequenceBuffer = [];

  frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
  statusText.textContent = 'Prediction stopped.';
});

// ---------- Convert landmarks → fixed-length array ----------
function flattenLandmarks(landmarks, expectedCount) {
  const arr = [];

   // Extract x, y, z for each landmark
  if (landmarks && landmarks.length) {
    for (const lm of landmarks) {
      arr.push(lm.x, lm.y, lm.z);
    }
  }

  // Pad with zeros if missing landmarks (important for model consistency)
  while (arr.length < expectedCount * 3) {
    arr.push(0, 0, 0);
  }

  // Ensure fixed size
  return arr.slice(0, expectedCount * 3);
}

// Match training format: pose(33*3) + left(21*3) + right(21*3) = 225
function extractKeypoints(results) {
  const pose = flattenLandmarks(results.poseLandmarks, 33);
  const leftHand = flattenLandmarks(results.leftHandLandmarks, 21);
  const rightHand = flattenLandmarks(results.rightHandLandmarks, 21);

  // Final feature vector = 225 values (matches training input)
  return [...pose, ...leftHand, ...rightHand];
}

// ---------- Send sequence to backend ----------
async function sendSequence(sequence) {
  if (isSending) return;

  try {
    isSending = true;
    statusText.textContent = 'Sending 30-frame sequence to backend...';

    // Send data to Flask backend
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence })
    });

    const result = await response.json();

    // Handle error response
    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    // Display prediction result
    predictionText.textContent = result.prediction || 'Unknown';
    confidenceText.textContent = `Confidence: ${
      result.confidence !== undefined ? result.confidence : '--'
    }`;
    statusText.textContent = 'Prediction received. Click "Start Prediction" for a new sign.';

    // Reset after prediction
    isPredicting = false;
    sequenceBuffer = [];
    frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
  } catch (err) {
    console.error(err);
    statusText.textContent = `Backend error: ${err.message}`;
  } finally {
    isSending = false;
  }
}

// ---------- Main loop: runs every frame ----------
holistic.onResults((results) => {
  // ---------- Draw current frame ----------
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Draw video frame onto canvas
  if (results.poseLandmarks) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#00BFFF',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.poseLandmarks, {
      color: '#FFD700',
      radius: 2
    });
  }

  // Draw left hand
  if (results.leftHandLandmarks) {
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.leftHandLandmarks, {
      color: '#FF0000',
      radius: 2
    });
  }

  // Draw right hand
  if (results.rightHandLandmarks) {
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
      color: '#9000ff',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.rightHandLandmarks, {
      color: '#FF0000',
      radius: 2
    });
  }

  ctx.restore();

  // ---------- Extract one frame's 225-dim feature ----------
  const keypoints = extractKeypoints(results);

  // Show current frame info
  jsonOutput.textContent = JSON.stringify({
    frame_index: frameIndex,
    feature_length: keypoints.length,
    feature_preview: keypoints.slice(0, 30)
  }, null, 2);

  // ---------- Sequence collection logic ----------
  if (isPredicting && !isSending) {
    sequenceBuffer.push(keypoints);

    // Keep only latest 30 frames (sliding window)
    if (sequenceBuffer.length > NUM_FRAMES) {
      sequenceBuffer.shift();
    }

    frameCountText.textContent = `Buffered frames: ${sequenceBuffer.length} / ${NUM_FRAMES}`;

    // Show progress
    if (sequenceBuffer.length < NUM_FRAMES) {
      statusText.textContent = `Collecting frames... ${sequenceBuffer.length}/${NUM_FRAMES}`;
    }

    // When 30 frames ready → send to backend
    if (sequenceBuffer.length === NUM_FRAMES) {
      sendSequence([...sequenceBuffer]);
    }
  }

  frameIndex++;
});

// ---------- Initialize camera ----------
const camera = new Camera(video, {
  onFrame: async () => {
    await holistic.send({ image: video });
  },
  width: 640,
  height: 480
});

// Start webcam
camera.start();