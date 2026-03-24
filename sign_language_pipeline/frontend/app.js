const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const jsonOutput = document.getElementById('jsonOutput');
const statusText = document.getElementById('status');
const frameCountText = document.getElementById('frameCount');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

canvas.width = 640;
canvas.height = 480;

const NUM_FRAMES = 30;
let frameIndex = 0;
let sequenceBuffer = [];
let isPredicting = false;
let isSending = false;

// ---------- Initial UI ----------
statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

// ---------- MediaPipe Holistic ----------
const holistic = new Holistic({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// ---------- Controls ----------
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

stopBtn.addEventListener('click', () => {
  isPredicting = false;
  isSending = false;
  sequenceBuffer = [];

  frameCountText.textContent = `Buffered frames: 0 / ${NUM_FRAMES}`;
  statusText.textContent = 'Prediction stopped.';
});

// ---------- Helper: flatten landmarks ----------
function flattenLandmarks(landmarks, expectedCount) {
  const arr = [];

  if (landmarks && landmarks.length) {
    for (const lm of landmarks) {
      arr.push(lm.x, lm.y, lm.z);
    }
  }

  while (arr.length < expectedCount * 3) {
    arr.push(0, 0, 0);
  }

  return arr.slice(0, expectedCount * 3);
}

// Match training format: pose(33*3) + left(21*3) + right(21*3) = 225
function extractKeypoints(results) {
  const pose = flattenLandmarks(results.poseLandmarks, 33);
  const leftHand = flattenLandmarks(results.leftHandLandmarks, 21);
  const rightHand = flattenLandmarks(results.rightHandLandmarks, 21);

  return [...pose, ...leftHand, ...rightHand];
}

// ---------- Send sequence to backend ----------
async function sendSequence(sequence) {
  if (isSending) return;

  try {
    isSending = true;
    statusText.textContent = 'Sending 30-frame sequence to backend...';

    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    predictionText.textContent = result.prediction || 'Unknown';
    confidenceText.textContent = `Confidence: ${
      result.confidence !== undefined ? result.confidence : '--'
    }`;
    statusText.textContent = 'Prediction received. Click "Start Prediction" for a new sign.';

    // 收到结果后自动停止，避免一直重复送同一段 sequence
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

holistic.onResults((results) => {
  // ---------- Draw current frame ----------
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Draw pose
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

  // ---------- Only collect after Start ----------
  if (isPredicting && !isSending) {
    sequenceBuffer.push(keypoints);

    if (sequenceBuffer.length > NUM_FRAMES) {
      sequenceBuffer.shift();
    }

    frameCountText.textContent = `Buffered frames: ${sequenceBuffer.length} / ${NUM_FRAMES}`;

    if (sequenceBuffer.length < NUM_FRAMES) {
      statusText.textContent = `Collecting frames... ${sequenceBuffer.length}/${NUM_FRAMES}`;
    }

    if (sequenceBuffer.length === NUM_FRAMES) {
      sendSequence([...sequenceBuffer]);
    }
  }

  frameIndex++;
});

// ---------- Camera ----------
const camera = new Camera(video, {
  onFrame: async () => {
    await holistic.send({ image: video });
  },
  width: 640,
  height: 480
});

camera.start();