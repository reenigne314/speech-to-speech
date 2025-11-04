import logging
import sys
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console

# --- Kokoro-specific imports ---
# This is the correct import based on your __init__.py
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)
console = Console()

# We need to map the incoming Whisper language codes
# to a specific Kokoro voice and language.
# Voice list: https://github.com/nazdridoy/kokoro-tts/blob/main/MODELS.md
WHISPER_LANGUAGE_TO_KOKORO_VOICE = {
    "en": "af_sarah",    # English (US) ***********to be edited
    "fr": "fr_fr_leonie",  # French
    "es": "es_es_leticia", # Spanish
    "zh": "zh_cn_nara",    # Chinese (Mandarin)
    "ja": "ja_jp_kaori",   # Japanese
    "ko": "ko_kr_soo",     # Korean
}

# The 'lang' parameter for kokoro.create *********to be edited
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "en-us", 
    "fr": "fr",
    "es": "es",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
}


class KokoroTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cpu",      # Kokoro ONNX uses 'cpu' or 'cuda'
        language="en",
        speaker_to_id="en", # This will be the key for the voice/lang maps
        speed=1.0,         # Kokoro supports a speed parameter
        gen_kwargs={},     # Unused
        blocksize=512,
        model_path="./kokoro-v1.0.onnx", # From your example
        voices_path="./voices-v1.0.bin", # From your example
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        self.blocksize = blocksize
        self.speed = speed
        self.target_sr = 16000 # The sample rate your pipeline expects

        # Map device to ONNX execution provider
        if self.device == "cuda":
            self.providers = ['CUDAExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']

        logger.info(f"Loading Kokoro model from {model_path} with provider: {self.providers}")
        
        # 1. Load the Kokoro model
        self.model = Kokoro(
            model_path=model_path,
            voices_path=voices_path,
        )
        
        # 2. Set the initial voice and lang
        self.voice = WHISPER_LANGUAGE_TO_KOKORO_VOICE[speaker_to_id]
        self.lang = WHISPER_LANGUAGE_TO_KOKORO_LANG[language]

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        try:
            # Generate audio but don't play it
            self._create_audio_recursive("Warming up.")
        except Exception as e:
            logger.error(f"KokoroTTSHandler warmup failed: {e}")

    def _create_audio_recursive(self, chunk: str, retry_count=0) -> tuple[list[float] | None, int | None]:
        """
        This is a server-side implementation of the `process_chunk_sequential`
        logic you provided. It handles the 'index 510 is out of bounds' error
        by recursively splitting the text.
        """
        try:
            # 1. This is the core synthesis function
            samples, sample_rate = self.model.create(
                chunk, 
                voice=self.voice, 
                speed=self.speed, 
                lang=self.lang
            )
            return samples, sample_rate
        
        except Exception as e:
            error_msg = str(e)
            
            # 2. Check for the specific long-text error
            if "index 510 is out of bounds" in error_msg:
                current_size = len(chunk)
                new_size = int(current_size * 0.6) # Reduce by 40%
                
                logger.warning(
                    f"Kokoro: Phoneme length error on chunk size {current_size}. "
                    f"Retrying with smaller pieces of size {new_size}."
                )
                
                # 3. Implement the splitting logic from the function you provided
                words = chunk.split()
                current_piece = []
                current_len = 0
                pieces = []
                
                for word in words:
                    word_len = len(word) + 1 # +1 for space
                    if current_len + word_len > new_size:
                        if current_piece:
                            pieces.append(' '.join(current_piece).strip())
                        current_piece = [word]
                        current_len = word_len
                    else:
                        current_piece.append(word)
                        current_len += word_len
                
                if current_piece:
                    pieces.append(' '.join(current_piece).strip())

                if not pieces:
                     logger.error(f"Kokoro: Failed to split chunk, aborting: {chunk}")
                     return None, None

                logger.debug(f"Split chunk into {len(pieces)} pieces.")

                # 4. Process each piece recursively
                all_samples = []
                last_sample_rate = None
                
                for i, piece in enumerate(pieces, 1):
                    logger.debug(f"Processing piece {i}/{len(pieces)}")
                    # Recursive call
                    samples, sr = self._create_audio_recursive(piece, retry_count + 1)
                    if samples:
                        all_samples.extend(samples)
                        last_sample_rate = sr
                
                if all_samples:
                    logger.debug(f"Successfully processed all {len(pieces)} pieces.")
                    return all_samples, last_sample_rate
                
                logger.error(f"Kokoro: Failed to process any pieces after splitting.")
                return None, None
            
            # 5. Handle other, unexpected errors
            else:
                # Log the error *without* writing to stdout
                logger.error(f"Error processing chunk: {e}")
                logger.error(f"DEBUG: Full error message: {error_msg}")
                logger.error(f"DEBUG: Chunk that failed: {chunk}")
                return None, None

    def process(self, llm_sentence):
        language_code = None

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Check if we need to switch languages
        if language_code is not None and self.language != language_code:
            try:
                self.voice = WHISPER_LANGUAGE_TO_KOKORO_VOICE[language_code]
                self.lang = WHISPER_LANGUAGE_TO_KOKORO_LANG[language_code]
                self.language = language_code
                logger.info(f"KokoroTTSHandler switched language to: {language_code} (Voice: {self.voice})")
            except KeyError:
                console.print(
                    f"[red]Language {language_code} not supported by Kokoro. Using {self.language} instead."
                )

        # 1. Generate the full audio clip using our new recursive helper
        audio_samples, orig_sr = self._create_audio_recursive(llm_sentence)
        
        if audio_samples is None or len(audio_samples) == 0 or not orig_sr:
            logger.error(f"KokoroTTSHandler failed to generate audio for: {llm_sentence}")
            self.should_listen.set()
            return # Must return, not yield

        # 2. Convert from list[float] to NumPy array
        audio_chunk = np.array(audio_samples, dtype=np.float32)

        # 3. Resample the audio to 16kHz (which your server pipeline expects)
        if orig_sr != self.target_sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=orig_sr, target_sr=self.target_sr)
        
        # 4. Convert from float32 to int16 (same as your Melo handler)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)

        # 5. Yield the audio in blocks (same as your Melo handler)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        # 6. Tell the VAD it can start listening again
        self.should_listen.set()