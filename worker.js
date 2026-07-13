/**
 * ಕನ್ನಡ Voice Editor — Web Worker
 * Handles model loading + transcription.
 * Designed to handle sequential chunk requests during live recording.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;

let transcriber = null;
let currentModelId = null;
let busy = false;
let queue = []; // queue chunks if one is already being processed

async function loadModel(modelId) {
  if (transcriber && currentModelId === modelId) {
    self.postMessage({ type: 'model-ready', modelId });
    return;
  }

  if (transcriber) {
    try { await transcriber.dispose(); } catch (_) {}
    transcriber = null;
  }

  self.postMessage({ type: 'model-loading', modelId });

  transcriber = await pipeline('automatic-speech-recognition', modelId, {
    dtype: 'fp32',
    device: 'wasm',
    progress_callback: (progress) => {
      self.postMessage({ type: 'model-progress', ...progress });
    },
  });

  currentModelId = modelId;
  self.postMessage({ type: 'model-ready', modelId });
}

async function transcribeChunk(audioData, chunkId, isFinal) {
  if (!transcriber) return;

  try {
    const isMultilingual = currentModelId.startsWith('onnx-community/');

    const opts = {};
    if (isMultilingual) {
      opts.language = 'kannada';
      opts.task = 'transcribe';
    }

    // Condition the decoder to output clean Kannada text
    opts.initial_prompt = 'ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ಬರೆಯಿರಿ. ಸರಿಯಾದ ವಿರಾಮ ಚಿಹ್ನೆಗಳನ್ನು ಬಳಸಿ.';

    const result = await transcriber(audioData, opts);
    const text = (result.text || '').trim();

    self.postMessage({
      type: 'transcription-chunk',
      text,
      chunkId,
      isFinal,
    });
  } catch (err) {
    self.postMessage({
      type: 'transcription-error',
      error: err.message || String(err),
      chunkId,
    });
  }
}

async function processQueue() {
  if (busy || queue.length === 0) return;
  busy = true;

  const { audioData, chunkId, isFinal } = queue.shift();
  await transcribeChunk(audioData, chunkId, isFinal);

  busy = false;
  processQueue(); // process next in queue
}

self.addEventListener('message', async (e) => {
  const { action, modelId, audioData, chunkId, isFinal } = e.data;

  switch (action) {
    case 'load-model':
      try { await loadModel(modelId); }
      catch (err) { self.postMessage({ type: 'model-error', error: err.message || String(err) }); }
      break;

    case 'transcribe-chunk':
      queue.push({ audioData, chunkId, isFinal });
      processQueue();
      break;
  }
});
