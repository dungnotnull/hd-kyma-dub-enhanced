"""HuggingFace model manager: lazy loading, CUDA auto-detect, idle unload."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "whisper":       "openai/whisper-large-v3",
    "xtts":          "coqui/XTTS-v2",
    "seamless_m4t":  "facebook/seamless-m4t-v2-large",
    "wav2lip":       "Rudrabha/Wav2Lip",
    "utmos":         "microsoft/UTMOS22",
    "minilm":        "sentence-transformers/all-MiniLM-L6-v2",
}

IDLE_TIMEOUT_S = 600


class HFModelManager:
    """Singleton lazy-loading registry for kyma-dub HuggingFace models."""

    _instance: Optional["HFModelManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, models_dir: str = "./models"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.models_dir = models_dir
        self.device = self._detect_device()
        self._models: dict = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        logger.info("HFModelManager initialized on device=%s", self.device)

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected")
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _reset_idle_timer(self, name: str):
        if name in self._timers:
            self._timers[name].cancel()
        timer = threading.Timer(IDLE_TIMEOUT_S, self._unload_model, args=[name])
        timer.daemon = True
        timer.start()
        self._timers[name] = timer

    def _unload_model(self, name: str):
        with self._lock:
            if name in self._models:
                del self._models[name]
                logger.info("Unloaded idle model: %s", name)
            if name in self._timers:
                del self._timers[name]

    def load_whisper(self):
        """Load openai/whisper-large-v3."""
        with self._lock:
            if "whisper" not in self._models:
                try:
                    import whisper as openai_whisper
                    self._models["whisper"] = openai_whisper.load_model(
                        "large-v3", download_root=self.models_dir
                    )
                    logger.info("Loaded openai/whisper-large-v3")
                except Exception as e:
                    logger.warning("whisper load failed (%s)", e)
                    self._models["whisper"] = None
            self._reset_idle_timer("whisper")
            return self._models["whisper"]

    def load_seamless_m4t(self):
        """Load facebook/seamless-m4t-v2-large."""
        with self._lock:
            if "seamless_m4t" not in self._models:
                try:
                    from transformers import AutoProcessor, SeamlessM4Tv2Model
                    processor = AutoProcessor.from_pretrained(
                        "facebook/seamless-m4t-v2-large",
                        cache_dir=self.models_dir,
                    )
                    model = SeamlessM4Tv2Model.from_pretrained(
                        "facebook/seamless-m4t-v2-large",
                        cache_dir=self.models_dir,
                    )
                    if self.device == "cuda":
                        model = model.to("cuda")
                    self._models["seamless_m4t"] = {"model": model, "processor": processor}
                    logger.info("Loaded facebook/seamless-m4t-v2-large on %s", self.device)
                except Exception as e:
                    logger.warning("SeamlessM4T load failed (%s)", e)
                    self._models["seamless_m4t"] = None
            self._reset_idle_timer("seamless_m4t")
            return self._models["seamless_m4t"]

    def load_xtts(self):
        """Load coqui/XTTS-v2."""
        with self._lock:
            if "xtts" not in self._models:
                try:
                    from TTS.api import TTS
                    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                    if self.device == "cuda":
                        model = model.to("cuda")
                    self._models["xtts"] = model
                    logger.info("Loaded coqui/XTTS-v2 on %s", self.device)
                except Exception as e:
                    logger.warning("XTTS-v2 load failed (%s)", e)
                    self._models["xtts"] = None
            self._reset_idle_timer("xtts")
            return self._models["xtts"]

    def load_minilm(self):
        """Load sentence-transformers/all-MiniLM-L6-v2."""
        with self._lock:
            if "minilm" not in self._models:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._models["minilm"] = SentenceTransformer(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        cache_folder=self.models_dir,
                    )
                    logger.info("Loaded all-MiniLM-L6-v2")
                except Exception as e:
                    logger.warning("MiniLM load failed (%s)", e)
                    self._models["minilm"] = None
            self._reset_idle_timer("minilm")
            return self._models["minilm"]

    def load_utmos(self):
        """Load microsoft/UTMOS22 MOS predictor."""
        with self._lock:
            if "utmos" not in self._models:
                try:
                    from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
                    processor = Wav2Vec2Processor.from_pretrained(
                        "microsoft/UTMOS22", cache_dir=self.models_dir
                    )
                    model = AutoModelForAudioClassification.from_pretrained(
                        "microsoft/UTMOS22", cache_dir=self.models_dir
                    )
                    if self.device == "cuda":
                        model = model.to("cuda")
                    self._models["utmos"] = {"model": model, "processor": processor}
                    logger.info("Loaded microsoft/UTMOS22 on %s", self.device)
                except Exception as e:
                    logger.warning("UTMOS22 load failed (%s)", e)
                    self._models["utmos"] = None
            self._reset_idle_timer("utmos")
            return self._models["utmos"]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts using MiniLM-L6-v2."""
        model = self.load_minilm()
        if model is None:
            return self._tfidf_fallback_encode(texts)
        try:
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.warning("MiniLM encode failed (%s)", e)
            return self._tfidf_fallback_encode(texts)

    def _tfidf_fallback_encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib, struct
        result = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            vec = [struct.unpack("f", h[i:i+4])[0] for i in range(0, 12, 4)]
            norm = sum(x**2 for x in vec) ** 0.5 or 1.0
            result.append([x / norm for x in vec])
        return result

    def preload_all(self):
        """Preload all models (for warm deployment)."""
        for name in ["minilm", "whisper", "utmos"]:
            try:
                getattr(self, f"load_{name}")()
            except Exception as e:
                logger.warning("Preload %s failed: %s", name, e)

    def get_model_info(self) -> dict:
        return {
            name: (model_id, name in self._models)
            for name, model_id in MODEL_REGISTRY.items()
        }
