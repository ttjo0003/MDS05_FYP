const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const statusText = document.getElementById('status');
const frameCountText = document.getElementById('frameCount');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

const progressFill = document.getElementById('progressFill');
const progressPercent = document.getElementById('progressPercent');
const overlayStatus = document.getElementById('overlayStatus');

const predictionHistoryBox = document.getElementById('predictionHistory');
const previousConfidenceDisplay = document.getElementById('previousConfidence');

canvas.width = 640;
canvas.height = 480;

const TARGET_FRAMES = 25;
const MAX_CAPTURE_FRAMES = 25;
const MIN_VALID_FRAMES = 5;
const BACKEND_URL = 'http://127.0.0.1:5000/predict';

let frameIndex = 0;
let rawSequenceBuffer = [];
let isPredicting = false;
let isSending = false;

let attemptCount = 0;
let predictedWords = [];
let latestCompletedConfidence = '--';

statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

function updateProgressUI(current, total) {
  const safeTotal = Math.max(total, 1);
  const percent = Math.min((current / safeTotal) * 100, 100);

  frameCountText.textContent = `Captured valid frames: ${current} / ${total}`;
  progressPercent.textContent = `${Math.round(percent)}%`;
  progressFill.style.width = `${percent}%`;
}

function renderPredictionHistory() {
  if (!predictionHistoryBox) return;

  if (predictedWords.length === 0) {
    predictionHistoryBox.innerHTML = `
      <div class="history-empty">No prediction history yet</div>
    `;
  } else {
    predictionHistoryBox.innerHTML = `
      <div class="history-item">
        <div class="history-word">${predictedWords.join(' ')}</div>
      </div>
    `;
  }

  if (previousConfidenceDisplay) {
    previousConfidenceDisplay.textContent = `Previous confidence: ${latestCompletedConfidence}`;
  }
}

function resetCaptureState() {
  rawSequenceBuffer = [];
  frameIndex = 0;
  updateProgressUI(0, MAX_CAPTURE_FRAMES);
}

function setIdleUI() {
  isPredicting = false;
  isSending = false;
  statusText.textContent = 'Prediction finished. Click "Start Prediction" to try again.';
  overlayStatus.textContent = 'Ready to start';
  startBtn.disabled = false;
  stopBtn.disabled = false;
}

function setRecordingUI() {
  predictionText.textContent = 'Recording...';
  confidenceText.textContent = 'Confidence: --';
  statusText.textContent = 'Recording gesture. Perform one complete sign, then click "Stop Prediction".';
  overlayStatus.textContent = 'Recording';
  startBtn.disabled = true;
  stopBtn.disabled = false;
  resetCaptureState();
}

function setProcessingUI() {
  statusText.textContent = 'Finalizing gesture sequence...';
  overlayStatus.textContent = 'Processing';
  startBtn.disabled = true;
  stopBtn.disabled = true;
}

updateProgressUI(0, MAX_CAPTURE_FRAMES);
renderPredictionHistory();

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
  const pose = [];
  const leftHand = flattenHandLandmarks(results.leftHandLandmarks, 21);
  const rightHand = flattenHandLandmarks(results.rightHandLandmarks, 21);

  // Full pose: 33 landmarks × 3 = 99
  if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
    for (const lm of results.poseLandmarks) {
      pose.push(lm.x, lm.y, lm.z);
    }
  }

  while (pose.length < 33 * 3) {
    pose.push(0, 0, 0);
  }

  // Total: 99 + 63 + 63 = 225
  return [...pose.slice(0, 99), ...leftHand, ...rightHand];
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
    const anyNonZero = frame.some((v) => Math.abs(v) > 1e-8);
    if (anyNonZero) count += 1;
  }
  return count;
}

function uniformSampleFrames(frames, targetCount = TARGET_FRAMES) {
  if (!Array.isArray(frames) || frames.length === 0) {
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

function drawUpperBodyPose(ctx, poseLandmarks) {
  if (!poseLandmarks || poseLandmarks.length < 33) return;

  const upperBodyIndices = Array.from({ length: 22 }, (_, i) => i + 11);

  const upperBodyConnections = POSE_CONNECTIONS.filter(([start, end]) =>
    start >= 11 && start <= 32 && end >= 11 && end <= 32
  );

  drawConnectors(ctx, poseLandmarks, upperBodyConnections, {
    color: '#00BFFF',
    lineWidth: 2
  });

  const upperBodyLandmarks = upperBodyIndices.map((i) => poseLandmarks[i]);

  drawLandmarks(ctx, upperBodyLandmarks, {
    color: '#FFD700',
    radius: 2
  });
}

async function sendSequence(sequence) {
  if (isSending) return;

  try {
    isSending = true;
    statusText.textContent = `Sending ${TARGET_FRAMES}-frame sequence to backend...`;
    overlayStatus.textContent = 'Sending';

    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence })
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    const finalPrediction = result.prediction || 'Unknown';
    const finalConfidence =
      result.confidence !== undefined && result.confidence !== null
        ? result.confidence
        : '--';

    predictionText.textContent = finalPrediction;
    confidenceText.textContent = `Confidence: ${finalConfidence}`;

    predictedWords.push(finalPrediction);
    latestCompletedConfidence = finalConfidence;
    renderPredictionHistory();

    statusText.textContent = result.message || 'Prediction received.';
    overlayStatus.textContent = 'Completed';
  } catch (err) {
    console.error(err);
    predictionText.textContent = 'Error';
    confidenceText.textContent = 'Confidence: --';
    statusText.textContent = `Backend error: ${err.message}`;
    overlayStatus.textContent = 'Error';
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
    overlayStatus.textContent = 'Too few frames';
    isPredicting = false;
    startBtn.disabled = false;
    stopBtn.disabled = false;
    return;
  }

  const sampledSequence = uniformSampleFrames(rawSequenceBuffer, TARGET_FRAMES);

  if (!sampledSequence || sampledSequence.length !== TARGET_FRAMES) {
    statusText.textContent = `Failed to build a valid ${TARGET_FRAMES}-frame sequence.`;
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    overlayStatus.textContent = 'Sampling failed';
    isPredicting = false;
    startBtn.disabled = false;
    stopBtn.disabled = false;
    return;
  }

  const nonZeroFrames = countNonZeroFrames(sampledSequence);
  if (nonZeroFrames < MIN_VALID_FRAMES) {
    statusText.textContent = `Too many empty frames after sampling (${nonZeroFrames}/${TARGET_FRAMES}). Please sign more clearly.`;
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    overlayStatus.textContent = 'Low-quality frames';
    isPredicting = false;
    startBtn.disabled = false;
    stopBtn.disabled = false;
    return;
  }

  await sendSequence(sampledSequence);
}

startBtn.addEventListener('click', () => {
  if (isSending) return;

  attemptCount += 1;
  isPredicting = true;

  // keep existing successful history visible during the new attempt
  renderPredictionHistory();

  setRecordingUI();
});

stopBtn.addEventListener('click', async () => {
  if (!isPredicting || isSending) return;

  isPredicting = false;
  setProcessingUI();
  await finalizePrediction();
});

holistic.onResults(async (results) => {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

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

  if (isPredicting && !isSending) {
    const validHands = hasValidHands(results);
    const validPose = hasValidPose(results);

    if (validHands) {
      rawSequenceBuffer.push(keypoints);
      updateProgressUI(rawSequenceBuffer.length, MAX_CAPTURE_FRAMES);

      if (validPose) {
        statusText.textContent = `Recording valid gesture frames... ${rawSequenceBuffer.length}/${MAX_CAPTURE_FRAMES}`;
        overlayStatus.textContent = 'Recording';
      } else {
        statusText.textContent = `Hand detected, pose weak. Recording... ${rawSequenceBuffer.length}/${MAX_CAPTURE_FRAMES}`;
        overlayStatus.textContent = 'Recording';
      }

      if (rawSequenceBuffer.length >= MAX_CAPTURE_FRAMES) {
        isPredicting = false;
        setProcessingUI();
        statusText.textContent = 'Max capture reached. Finalizing automatically...';
        overlayStatus.textContent = 'Auto finalizing';
        await finalizePrediction();
      }
    } else {
      statusText.textContent = 'No hands detected. Keep your signing hand(s) visible.';
      overlayStatus.textContent = 'Waiting for hands';
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