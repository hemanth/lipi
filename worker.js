/**
 * Lipi Indic Voice Editor — Web Worker
 * Handles Whisper speech recognition (Kannada) + Paar Indic Translation (12 languages).
 * Preserves Whisper transcription pipeline while providing fast, cached Indic translation.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

// Force remote fetch from Hugging Face Hub (browser environment)
env.allowLocalModels = false;

const ctx = typeof self !== 'undefined' ? self : globalThis;

// ─── Whisper Speech Recognition State ───
let transcriber = null;
let currentWhisperModelId = null;

// ─── Translation Pipelines Cache ───
// Map of modelId -> pipeline instance
const translationPipelines = new Map();
// Map of modelId -> in-flight loading Promise
const loadingPipelines = new Map();

// ─── Concurrency & Queue Management ───
// Dual queues prevent starvation: audio chunks and translation requests interleave fairly
const audioQueue = [];
const translationQueue = [];
let isProcessing = false;
let lastServed = 'audio'; // 'audio' | 'translation'

// ═══════════════════════════════════════════
//  Whisper Speech Recognition Functions
// ═══════════════════════════════════════════

async function loadWhisperModel(modelId) {
  if (transcriber && currentWhisperModelId === modelId) {
    ctx.postMessage({ type: 'model-ready', modelId });
    return;
  }

  if (transcriber) {
    try { await transcriber.dispose(); } catch (_) {}
    transcriber = null;
  }

  ctx.postMessage({ type: 'model-loading', modelId });

  transcriber = await pipeline('automatic-speech-recognition', modelId, {
    dtype: 'fp32',
    device: 'wasm',
    progress_callback: (progress) => {
      ctx.postMessage({ type: 'model-progress', ...progress });
    },
  });

  currentWhisperModelId = modelId;
  ctx.postMessage({ type: 'model-ready', modelId });
}

async function transcribeChunk(audioData, chunkId, isFinal) {
  if (!transcriber) return;

  try {
    const opts = {
      // Condition the decoder to output clean Kannada text
      initial_prompt: 'ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿ ಸ್ಪಷ್ಟವಾಗಿ ಬರೆಯಿರಿ.',
    };

    const result = await transcriber(audioData, opts);
    let text = (result.text || '').trim();

    // Filter garbage: repetitive patterns, non-Kannada hallucinations
    if (isGarbage(text)) {
      text = '';
    }

    ctx.postMessage({
      type: 'transcription-chunk',
      text,
      chunkId,
      isFinal,
    });
  } catch (err) {
    ctx.postMessage({
      type: 'transcription-error',
      error: err.message || String(err),
      chunkId,
    });
  }
}

/**
 * Detect garbage/hallucinated output from Whisper (Kannada only).
 * Common patterns: repeated phrases, non-Kannada text when expecting Kannada,
 * single repeated characters, or very long repetitive sequences.
 */
function isGarbage(text) {
  if (!text || text.length < 2) return false;

  // Detect repeated short phrases (e.g. "ನಮಸ್ಕಾರ ನಮಸ್ಕಾರ ನಮಸ್ಕಾರ")
  const words = text.split(/\s+/);
  if (words.length >= 4) {
    const unique = new Set(words);
    if (unique.size <= 2) return true; // same 1-2 words repeated 4+ times
  }

  // Detect character-level repetition (e.g. "aaaaaaa" or "।।।।।।")
  if (/(.)\1{6,}/.test(text)) return true;

  // Detect repeated bigrams/trigrams filling the output
  if (words.length >= 6) {
    const bigrams = [];
    for (let i = 0; i < words.length - 1; i++) bigrams.push(words[i] + ' ' + words[i + 1]);
    const uniqueBigrams = new Set(bigrams);
    if (uniqueBigrams.size === 1) return true;
  }

  // Mostly non-Kannada when we expect Kannada (Latin/CJK hallucination)
  const kannadaChars = (text.match(/[\u0C80-\u0CFF]/g) || []).length;
  const totalAlpha = (text.match(/[^\s\d.,!?;:'"()\-]/g) || []).length;
  if (totalAlpha > 5 && kannadaChars / totalAlpha < 0.3) return true;

  return false;
}

// ═══════════════════════════════════════════
//  Paar Translation Functions
// ═══════════════════════════════════════════

function resolvePaarModelId(language, modelId) {
  if (modelId) return modelId;
  if (language) return `udgamlabs/Paar-en-${language}`;
  return null;
}

async function getTranslationPipeline(modelId, language) {
  if (translationPipelines.has(modelId)) {
    return translationPipelines.get(modelId);
  }

  if (loadingPipelines.has(modelId)) {
    return await loadingPipelines.get(modelId);
  }

  const loadPromise = (async () => {
    try {
      ctx.postMessage({
        type: 'model-loading',
        language,
        modelId,
      });

      const translator = await pipeline('translation', modelId, {
        dtype: 'q8',
        progress_callback: (progress) => {
          ctx.postMessage({
            type: 'model-progress',
            language,
            modelId,
            progress: progress.progress ?? (progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0),
            ...progress,
          });
        },
      });

      translationPipelines.set(modelId, translator);
      ctx.postMessage({
        type: 'model-ready',
        language,
        modelId,
      });
      return translator;
    } finally {
      loadingPipelines.delete(modelId);
    }
  })();

  loadingPipelines.set(modelId, loadPromise);
  return await loadPromise;
}

async function loadTranslationModel(language, modelId) {
  const resolvedModelId = resolvePaarModelId(language, modelId);
  if (!resolvedModelId) {
    throw new Error(`Missing modelId or language for translation model loading`);
  }

  if (translationPipelines.has(resolvedModelId)) {
    ctx.postMessage({
      type: 'model-ready',
      language,
      modelId: resolvedModelId,
    });
    return translationPipelines.get(resolvedModelId);
  }

  return await getTranslationPipeline(resolvedModelId, language);
}

async function executeTranslation({ text, targetLang, id, modelId }) {
  if (!text || !text.trim()) {
    ctx.postMessage({
      type: 'translation-result',
      id,
      text: text || '',
      targetLang,
    });
    return;
  }

  try {
    const resolvedModelId = resolvePaarModelId(targetLang, modelId);
    if (!resolvedModelId) {
      throw new Error(`Cannot resolve model ID for target language: ${targetLang}`);
    }

    const translator = await getTranslationPipeline(resolvedModelId, targetLang);

    // Greedy decoding with Paar configuration
    const output = await translator(text, {
      num_beams: 1,
      max_new_tokens: 200,
      decoder_start_token_id: 2,
    });

    let translatedText = '';
    if (Array.isArray(output) && output.length > 0) {
      const first = output[0];
      if (typeof first === 'string') {
        translatedText = first;
      } else if (first && typeof first === 'object') {
        translatedText = first.translation_text ?? first.generated_text ?? '';
      }
    } else if (output && typeof output === 'object') {
      translatedText = output.translation_text ?? output.generated_text ?? '';
    } else if (typeof output === 'string') {
      translatedText = output;
    }

    ctx.postMessage({
      type: 'translation-result',
      id,
      text: translatedText.trim(),
      targetLang,
    });
  } catch (err) {
    ctx.postMessage({
      type: 'translation-error',
      id,
      targetLang,
      error: err.message || String(err),
    });
  }
}

// ═══════════════════════════════════════════
//  Fair Round-Robin Concurrency Scheduler
// ═══════════════════════════════════════════

function scheduleNext() {
  if (isProcessing) return;

  let nextTask = null;
  let taskType = null;

  if (translationQueue.length > 0 && audioQueue.length > 0) {
    if (lastServed === 'audio') {
      nextTask = translationQueue.shift();
      taskType = 'translation';
      lastServed = 'translation';
    } else {
      nextTask = audioQueue.shift();
      taskType = 'audio';
      lastServed = 'audio';
    }
  } else if (translationQueue.length > 0) {
    nextTask = translationQueue.shift();
    taskType = 'translation';
    lastServed = 'translation';
  } else if (audioQueue.length > 0) {
    nextTask = audioQueue.shift();
    taskType = 'audio';
    lastServed = 'audio';
  } else {
    return;
  }

  isProcessing = true;

  (async () => {
    try {
      if (taskType === 'audio') {
        await transcribeChunk(nextTask.audioData, nextTask.chunkId, nextTask.isFinal);
      } else if (taskType === 'translation') {
        await executeTranslation(nextTask);
      }
    } catch (err) {
      console.error(`[worker] Unexpected execution error in ${taskType} task:`, err);
    } finally {
      isProcessing = false;
      scheduleNext();
    }
  })();
}

function enqueueAudioChunk(item) {
  audioQueue.push(item);
  scheduleNext();
}

function enqueueTranslation(item) {
  translationQueue.push(item);
  scheduleNext();
}

// ═══════════════════════════════════════════
//  Message Event Listener & Router
// ═══════════════════════════════════════════

ctx.addEventListener('message', async (e) => {
  const data = e.data || {};
  const { action, modelId, audioData, chunkId, isFinal, language, targetLang, text, id } = data;

  switch (action) {
    // ─── Whisper Actions ───
    case 'load-model':
      try {
        await loadWhisperModel(modelId);
      } catch (err) {
        ctx.postMessage({
          type: 'model-error',
          error: err.message || String(err),
        });
      }
      break;

    case 'transcribe-chunk':
      enqueueAudioChunk({ audioData, chunkId, isFinal });
      break;

    // ─── Paar Translation Actions ───
    case 'load-translation-model':
      try {
        await loadTranslationModel(language, modelId);
      } catch (err) {
        ctx.postMessage({
          type: 'model-error',
          language,
          modelId,
          error: err.message || String(err),
        });
        ctx.postMessage({
          type: 'translation-error',
          language,
          modelId,
          error: err.message || String(err),
        });
      }
      break;

    case 'translate':
      enqueueTranslation({ text, targetLang, id, modelId });
      break;

    default:
      console.warn(`[worker] Unknown action: ${action}`);
      break;
  }
});
