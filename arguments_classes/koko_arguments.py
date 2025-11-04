from dataclasses import dataclass, field


@dataclass
class KokoroTTSHandlerArguments:
    kokoro_device: str = field(
        default="cpu",
        metadata={
            "help": "The device to be used for speech synthesis (e.g., 'cpu', 'cuda'). Default is 'cpu'."
        },
    )
    kokoro_language: str = field(
        default="en",
        metadata={
            "help": "The language of the text to be synthesized (e.g., 'en', 'es', 'fr'). Default is 'en'."
        },
    )
    kokoro_speaker_to_id: str = field(
        default="en",
        metadata={
            "help": "The speaker key to use for voice mapping. Corresponds to WHISPER_LANGUAGE_TO_KOKORO_VOICE keys. Default is 'en'."
        },
    )
    kokoro_speed: float = field(
        default=1.0,
        metadata={"help": "The speed of the speech synthesis. Default is 1.0."},
    )
    kokoro_blocksize: int = field(
        default=512,
        metadata={"help": "The audio chunk size for yielding. Default is 512."},
    )
    kokoro_model_path: str = field(
        default="./kokoro-v1.0.onnx",
        metadata={"help": "Path to the kokoro-v1.0.onnx model file."},
    )
    kokoro_voices_path: str = field(
        default="./voices-v1.0.bin",
        metadata={"help": "Path to the voices-v1.0.bin file."},
    )