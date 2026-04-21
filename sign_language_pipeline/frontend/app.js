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

const TARGET_FRAMES = 30;
const MAX_CAPTURE_FRAMES = 30;
const MIN_VALID_FRAMES = 20;
const BACKEND_URL = 'http://127.0.0.1:5000/predict';

let frameIndex = 0;
let rawSequenceBuffer = [];
let isPredicting = false;
let isSending = false;

statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
frameCountText.textContent = `Captured valid frames: 0 / ${MAX_CAPTURE_FRAMES}`;
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

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

function flattenHandLandmarks(landmarks, expectedCount) {
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

function extractKeypoints(results) {
  let pose = new Array(88).fill(0); // 22 landmarks * 4 values
  const leftHand = flattenHandLandmarks(results.leftHandLandmarks, 21);   // 63
  const rightHand = flattenHandLandmarks(results.rightHandLandmarks, 21); // 63

  if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
    const upperBodyPose = results.poseLandmarks.slice(11, 33); // landmarks 11-32

    const poseArr = [];
    for (const lm of upperBodyPose) {
      poseArr.push(lm.x, lm.y, lm.z, lm.visibility ?? 0);
    }

    if (poseArr.length === 88) {
      pose = poseArr;
    }
  }

  return [...pose, ...leftHand, ...rightHand]; // total 214
}

function hasValidHands(results) {
  const leftValid = results.leftHandLandmarks && results.leftHandLandmarks.length > 0;
  const rightValid = results.rightHandLandmarks && results.rightHandLandmarks.length > 0;
  return !!(leftValid || rightValid);
}

function hasValidPose(results) {
  return !!(results.poseLandmarks && results.poseLandmarks.length > 0);
}

function countNonZeroFrames(sequence) {
  let count = 0;
  for (const frame of sequence) {
    const anyNonZero = frame.some(v => Math.abs(v) > 1e-8);
    if (anyNonZero) count += 1;
  }
  return count;
}

function uniformSampleFrames(frames, targetCount = TARGET_FRAMES) {
  if (!Array.isArray(frames) || frames.length < targetCount) {
    return null;
  }

  if (frames.length === targetCount) {
    return [...frames];
  }

  const sampled = [];
  for (let i = 0; i < targetCount; i++) {
    const idx = Math.round((i * (frames.length - 1)) / (targetCount - 1));
    sampled.push(frames[idx]);
  }
  return sampled;
}

function resetCaptureState() {
  rawSequenceBuffer = [];
  frameIndex = 0;
  frameCountText.textContent = `Captured valid frames: 0 / ${MAX_CAPTURE_FRAMES}`;
}

function setIdleUI() {
  isPredicting = false;
  isSending = false;
  statusText.textContent = 'Prediction finished. Click "Start Prediction" to try again.';
  frameCountText.textContent = `Captured valid frames: 0 / ${MAX_CAPTURE_FRAMES}`;
}

function drawUpperBodyPose(ctx, poseLandmarks) {
  if (!poseLandmarks || poseLandmarks.length < 33) return;

  const upperBodyIndices = Array.from({ length: 22 }, (_, i) => i + 11);

  const upperBodyConnections = POSE_CONNECTIONS.filter(([start, end]) =>
    start >= 11 && start <= 32 && end >= 11 && end <= 32
  );

  drawConnectors(
    ctx,
    poseLandmarks,
    upperBodyConnections,
    {
      color: '#00BFFF',
      lineWidth: 2
    }
  );

  const upperBodyLandmarks = upperBodyIndices.map(i => poseLandmarks[i]);

  drawLandmarks(ctx, upperBodyLandmarks, {
    color: '#FFD700',
    radius: 2
  });
}

async function sendSequence(sequence) {
  if (isSending) return;

  try {
    isSending = true;
    statusText.textContent = 'Sending sampled 30-frame sequence to backend...';

    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    predictionText.textContent = result.prediction || 'Unknown';

    if (result.confidence !== undefined && result.confidence !== null) {
      confidenceText.textContent = `Confidence: ${result.confidence}`;
    } else {
      confidenceText.textContent = 'Confidence: --';
    }

    if (result.top_k && Array.isArray(result.top_k)) {
      jsonOutput.textContent = JSON.stringify({
        sampled_frames: sequence.length,
        backend_result: result
      }, null, 2);
    }

    statusText.textContent = result.message || 'Prediction received.';
  } catch (err) {
    console.error(err);
    statusText.textContent = `Backend error: ${err.message}`;
  } finally {
    resetCaptureState();
    setIdleUI();
  }
}

async function finalizePrediction() {
  if (isSending) return;

  if (rawSequenceBuffer.length < MIN_VALID_FRAMES) {
    statusText.textContent = `Not enough valid gesture frames. Need at least ${MIN_VALID_FRAMES}, got ${rawSequenceBuffer.length}.`;
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    resetCaptureState();
    isPredicting = false;
    return;
  }

  const sampledSequence = uniformSampleFrames(rawSequenceBuffer, TARGET_FRAMES);

  if (!sampledSequence || sampledSequence.length !== TARGET_FRAMES) {
    statusText.textContent = 'Failed to build a valid 30-frame sequence.';
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    resetCaptureState();
    isPredicting = false;
    return;
  }

  const nonZeroFrames = countNonZeroFrames(sampledSequence);
  if (nonZeroFrames < MIN_VALID_FRAMES) {
    statusText.textContent = `Too many empty frames after sampling (${nonZeroFrames}/${TARGET_FRAMES}). Please sign more clearly.`;
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    resetCaptureState();
    isPredicting = false;
    return;
  }

  await sendSequence(sampledSequence);
}

startBtn.addEventListener('click', () => {
  if (isSending) return;

  isPredicting = true;
  resetCaptureState();

  predictionText.textContent = 'Recording...';
  confidenceText.textContent = 'Confidence: --';
  statusText.textContent = 'Recording gesture. Perform one complete sign, then click "Stop Prediction".';
});

stopBtn.addEventListener('click', async () => {
  if (!isPredicting || isSending) return;

  statusText.textContent = 'Finalizing gesture sequence...';
  isPredicting = false;
  await finalizePrediction();
});

holistic.onResults(async (results) => {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.poseLandmarks) {
  drawUpperBodyPose(ctx, results.poseLandmarks);
  }

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

  const keypoints = extractKeypoints(results);

  jsonOutput.textContent = JSON.stringify({
    frame_index: frameIndex,
    feature_length: keypoints.length,
    pose_range: '0:88',
    left_hand_range: '88:151',
    right_hand_range: '151:214',
    left_hand_detected: !!(results.leftHandLandmarks && results.leftHandLandmarks.length),
    right_hand_detected: !!(results.rightHandLandmarks && results.rightHandLandmarks.length),
    pose_detected: !!(results.poseLandmarks && results.poseLandmarks.length),
    valid_buffer_frames: rawSequenceBuffer.length,
    feature_preview: keypoints.slice(0, 30)
  }, null, 2);

  if (isPredicting && !isSending) {
    const validHands = hasValidHands(results);
    const validPose = hasValidPose(results);

    if (validHands) {
      rawSequenceBuffer.push(keypoints);

      frameCountText.textContent = `Captured valid frames: ${rawSequenceBuffer.length} / ${MAX_CAPTURE_FRAMES}`;

      if (validPose) {
        statusText.textContent = `Recording valid gesture frames... ${rawSequenceBuffer.length}/${MAX_CAPTURE_FRAMES}`;
      } else {
        statusText.textContent = `Hand detected, pose weak. Recording... ${rawSequenceBuffer.length}/${MAX_CAPTURE_FRAMES}`;
      }

      if (rawSequenceBuffer.length >= MAX_CAPTURE_FRAMES) {
        isPredicting = false;
        statusText.textContent = 'Max capture reached. Finalizing automatically...';
        await finalizePrediction();
      }
    } else {
      statusText.textContent = 'No hands detected. Keep your signing hand(s) visible.';
    }
  }

  frameIndex++;
});

const camera = new Camera(video, {
  onFrame: async () => {
    await holistic.send({ image: video });
  },
  width: 640,
  height: 480
});

camera.start();