// Ragdoll judge — inference runs entirely in the browser via onnxruntime-web.
// The preprocessing below must mirror the Python eval pipeline exactly:
// Resize(256) -> CenterCrop(224) -> scale to [0,1] -> ImageNet normalize.

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CROP = 224;
const RESIZE = 256;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const placeholder = document.getElementById('photo-placeholder');
const preview = document.getElementById('preview');
const judgeBtn = document.getElementById('judge-btn');
const result = document.getElementById('result');
const verdict = document.getElementById('verdict');
const verdictSub = document.getElementById('verdict-sub');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const modelStatus = document.getElementById('model-status');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');

let session = null;
let currentImage = null;

loadModel();

async function loadModel() {
  try {
    // Fetch manually so the 43 MB download shows real progress.
    const resp = await fetch('model.onnx');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const total = Number(resp.headers.get('Content-Length')) || 0;
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) progressFill.style.width = `${(received / total) * 100}%`;
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    session = await ort.InferenceSession.create(bytes.buffer, {
      executionProviders: ['wasm'],
    });
    modelStatus.classList.add('ready');
    statusText.textContent = 'judge ready — drop a cat';
    updateJudgeState();
  } catch (err) {
    statusText.textContent = `couldn't load the model — try refreshing (${err.message})`;
  }
}

// --- upload handling ---

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    dropzone.classList.add('error');
    setTimeout(() => dropzone.classList.remove('error'), 400);
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  try {
    await img.decode();
  } catch {
    dropzone.classList.add('error');
    setTimeout(() => dropzone.classList.remove('error'), 400);
    return;
  }
  currentImage = img;
  preview.src = url;
  placeholder.classList.add('hidden');
  preview.classList.remove('hidden');
  result.classList.add('hidden');
  updateJudgeState();
}

function updateJudgeState() {
  judgeBtn.disabled = !(session && currentImage);
}

// --- inference ---

judgeBtn.addEventListener('click', async () => {
  judgeBtn.disabled = true;
  judgeBtn.textContent = 'judging…';
  try {
    const tensor = preprocess(currentImage);
    const feeds = { [session.inputNames[0]]: tensor };
    const output = await session.run(feeds);
    const logits = output[session.outputNames[0]].data; // [not-ragdoll, ragdoll]
    showVerdict(logits);
  } catch (err) {
    verdict.textContent = 'the judge fumbled';
    verdictSub.textContent = err.message;
    result.className = 'result not-ragdoll';
  } finally {
    judgeBtn.textContent = 'Judge my cat';
    updateJudgeState();
  }
});

function preprocess(img) {
  // Resize so the SHORTER side is 256 (aspect preserved), like torchvision Resize(256).
  const scale = RESIZE / Math.min(img.naturalWidth, img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const resized = document.createElement('canvas');
  resized.width = w;
  resized.height = h;
  const rctx = resized.getContext('2d');
  rctx.imageSmoothingEnabled = true;
  rctx.imageSmoothingQuality = 'high';
  rctx.drawImage(img, 0, 0, w, h);

  // Center crop 224x224.
  const crop = document.createElement('canvas');
  crop.width = CROP;
  crop.height = CROP;
  const cctx = crop.getContext('2d');
  cctx.drawImage(
    resized,
    Math.floor((w - CROP) / 2), Math.floor((h - CROP) / 2), CROP, CROP,
    0, 0, CROP, CROP,
  );

  // RGBA pixels -> normalized float32 CHW tensor.
  const rgba = cctx.getImageData(0, 0, CROP, CROP).data;
  const chw = new Float32Array(3 * CROP * CROP);
  const plane = CROP * CROP;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      chw[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor('float32', chw, [1, 3, CROP, CROP]);
}

function showVerdict(logits) {
  // Stable softmax over the two logits for a confidence number.
  const m = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - m);
  const e1 = Math.exp(logits[1] - m);
  const pRagdoll = e1 / (e0 + e1);
  const isRagdoll = pRagdoll >= 0.5;
  const confidence = Math.round((isRagdoll ? pRagdoll : 1 - pRagdoll) * 100);

  result.className = `result ${isRagdoll ? 'ragdoll' : 'not-ragdoll'}`;
  verdict.textContent = isRagdoll
    ? 'Goated cat breed. 🐾'
    : "Your cat gets mogged daily. It's not a ragdoll!";
  verdictSub.textContent = isRagdoll
    ? 'Certified ragdoll.'
    : 'Still a good cat. Just not THE cat.';
  confidenceFill.style.width = `${confidence}%`;
  confidenceText.textContent = `the judge is ${confidence}% sure`;
}
