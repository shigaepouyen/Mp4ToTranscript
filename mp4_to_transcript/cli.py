from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which
from typing import Any, Sequence

try:
    from rich.console import Console as RichConsole
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ImportError:
    RichConsole = None
    Progress = None
    BarColumn = SpinnerColumn = TextColumn = TimeElapsedColumn = None  # type: ignore[assignment]


SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
DEFAULT_MODEL = "large"
DEFAULT_COMBINED_BASENAME = "transcription_complete"
DEFAULT_OUTPUT_FORMAT = "txt"
SUPPORTED_OUTPUT_FORMATS = ("txt", "md", "both")
OUTPUT_SUFFIXES = {"txt": ".txt", "md": ".md", "both": ".md"}
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
DEFAULT_LLM_PROVIDER = "none"
DEFAULT_LLM_MODEL = "gpt-5-mini"
OPENAI_API_KEY_ENV_VARS = ("OPENAI_API_KEY",)
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
MEETING_REPORT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
MARKUP_RE = re.compile(r"\[/?[^\]]+\]")
DISFLUENCY_RE = re.compile(
    r"(?i)(?:(?<=^)|(?<=[\s,;:]))(?:euh|heu|hum|bah|ben|hein|bon ben|ouais ouais)(?=$|[\s,;:.!?])"
)
WORD_REPETITION_RE = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,;:.!?])")
MISSING_SPACE_AFTER_PUNCTUATION_RE = re.compile(r"([,;:.!?])(?=[^\s,;:.!?])")
SPEAKER_TOKEN_RE = re.compile(r"(?i)^(?:speaker|spk|intervenant)[ _-]?0*(\d+)$")


class PlainConsole:
    def print(self, message: object = "") -> None:
        print(MARKUP_RE.sub("", str(message)))


console = RichConsole() if RichConsole is not None else PlainConsole()


def load_whisper_module():
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Le package `openai-whisper` n'est pas installe. "
            "Installe-le avec `python3 -m pip install -U openai-whisper`."
        ) from exc

    if not hasattr(whisper, "load_model"):
        module_file = getattr(whisper, "__file__", "inconnu")
        raise RuntimeError(
            "Le module `whisper` charge actuellement un mauvais package et pas `openai-whisper`.\n"
            f"Module detecte: {module_file}\n"
            "Corrige avec:\n"
            "  python3 -m pip uninstall -y whisper\n"
            "  python3 -m pip install -U openai-whisper"
        )

    return whisper


def ensure_ffmpeg_available() -> None:
    missing = [binary for binary in ("ffmpeg", "ffprobe") if which(binary) is None]
    if missing:
        raise RuntimeError(
            "Dependance manquante: "
            + ", ".join(missing)
            + ". Installe FFmpeg pour traiter les fichiers audio/video."
        )


def collect_input_files(input_path: Path, recursive: bool = False) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chemin introuvable: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extension non supportee: {input_path.suffix}. "
                f"Extensions acceptees: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return [input_path]

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        scope = "recursivement" if recursive else "au premier niveau"
        raise FileNotFoundError(
            f"Aucun fichier audio/video supporte trouve {scope} dans {input_path}. "
            f"Extensions cherchees: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return files


def probe_duration_seconds(file_path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    raw_duration = result.stdout.strip()
    if not raw_duration:
        return None

    try:
        return float(raw_duration)
    except ValueError:
        return None


def estimate_total_duration_seconds(files: Sequence[Path]) -> float:
    total = 0.0
    for file_path in files:
        duration = probe_duration_seconds(file_path)
        if duration is not None:
            total += duration
    return total


def format_duration(total_seconds: float) -> str:
    rounded = max(0, int(total_seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)

    chunks = []
    if hours:
        chunks.append(f"{hours}h")
    if minutes or hours:
        chunks.append(f"{minutes:02d}m")
    chunks.append(f"{seconds:02d}s")
    return " ".join(chunks)


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def default_combined_filename(output_format: str) -> str:
    return f"{DEFAULT_COMBINED_BASENAME}{OUTPUT_SUFFIXES[output_format]}"


def is_supported_output_path(path: Path) -> bool:
    return path.suffix.lower() in OUTPUT_SUFFIXES.values()


def normalize_combined_name(name: str, output_format: str = DEFAULT_OUTPUT_FORMAT) -> str:
    expected_suffix = OUTPUT_SUFFIXES[output_format]
    current_suffix = Path(name).suffix.lower()
    if current_suffix in OUTPUT_SUFFIXES.values() and current_suffix != expected_suffix:
        raise ValueError(
            f"Le nom de sortie `{name}` ne correspond pas au format `{output_format}` attendu ({expected_suffix})."
        )
    return name if name.lower().endswith(expected_suffix) else f"{name}{expected_suffix}"


def normalize_transcript_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def cleanup_transcript_text(text: str) -> str:
    cleaned = normalize_transcript_text(text)
    cleaned = DISFLUENCY_RE.sub(" ", cleaned)
    cleaned = WORD_REPETITION_RE.sub(r"\1", cleaned)
    cleaned = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", cleaned)
    cleaned = MISSING_SPACE_AFTER_PUNCTUATION_RE.sub(r"\1 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:")

    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_speaker_label(raw_label: object) -> str | None:
    if raw_label is None:
        return None

    label = normalize_transcript_text(str(raw_label))
    if not label:
        return None

    match = SPEAKER_TOKEN_RE.match(label)
    if match:
        return f"Intervenant {int(match.group(1)) + 1}"

    return label.replace("_", " ")


def extract_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw_segment in result.get("segments") or []:
        text = normalize_transcript_text(str(raw_segment.get("text", "")))
        if not text:
            continue

        segments.append(
            {
                "start": float(raw_segment.get("start", 0.0)),
                "end": float(raw_segment.get("end", 0.0)),
                "text": text,
                "speaker": normalize_speaker_label(
                    raw_segment.get("speaker")
                    or raw_segment.get("speaker_id")
                    or raw_segment.get("speaker_label")
                ),
            }
        )
    return segments


def collect_detected_speakers(segments: Sequence[dict[str, Any]]) -> list[str]:
    speakers: list[str] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if speaker and speaker not in speakers:
            speakers.append(speaker)
    return speakers


def maybe_cleanup_text(text: str, cleanup_mode: str) -> str:
    return cleanup_transcript_text(text) if cleanup_mode in {"clean", "meeting"} else normalize_transcript_text(text)


def find_huggingface_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token.strip() or None

    for env_name in DEFAULT_HF_TOKEN_ENV_VARS:
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()

    return None


def clone_result_with_segments(result: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(result)
    cloned["segments"] = [dict(segment) for segment in result.get("segments") or []]
    return cloned


def build_diarization_options(min_speakers: int | None, max_speakers: int | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if min_speakers is not None:
        options["min_speakers"] = min_speakers
    if max_speakers is not None:
        options["max_speakers"] = max_speakers
    return options


def load_pyannote_pipeline(model_name: str, auth_token: str, device: str) -> Any:
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "La diarisation demande `pyannote.audio`, non installe.\n"
            "Installe-la avec `python3 -m pip install pyannote.audio` puis relance avec un token Hugging Face."
        ) from exc

    pipeline = Pipeline.from_pretrained(model_name, token=auth_token)

    if device != "cpu":
        try:
            import torch

            pipeline.to(torch.device(device))
        except Exception:
            console.print(f"[yellow]Impossible d'envoyer la diarisation sur {device}, execution sur cpu.[/yellow]")

    return pipeline


def diarization_annotation_to_spans(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None:
        return []

    spans: list[dict[str, Any]] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            label = normalize_speaker_label(speaker)
            if label is None:
                continue
            spans.append({"start": float(turn.start), "end": float(turn.end), "speaker": label})
        return spans

    for item in annotation:
        if len(item) == 2:
            turn, speaker = item
        elif len(item) == 3:
            turn, _, speaker = item
        else:
            continue
        label = normalize_speaker_label(speaker)
        if label is None:
            continue
        spans.append({"start": float(turn.start), "end": float(turn.end), "speaker": label})
    return spans


def compute_segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers_to_segments(
    segments: Sequence[dict[str, Any]],
    speaker_spans: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not segments or not speaker_spans:
        return [dict(segment) for segment in segments]

    assigned_segments: list[dict[str, Any]] = []
    for segment in segments:
        assigned_segment = dict(segment)
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        best_speaker = assigned_segment.get("speaker")
        best_overlap = 0.0

        for span in speaker_spans:
            overlap = compute_segment_overlap(seg_start, seg_end, float(span["start"]), float(span["end"]))
            if overlap <= best_overlap:
                continue
            best_overlap = overlap
            best_speaker = span["speaker"]

        if best_speaker:
            assigned_segment["speaker"] = best_speaker
        assigned_segments.append(assigned_segment)

    return assigned_segments


def apply_speaker_diarization(
    result: dict[str, Any],
    file_path: Path,
    diarization_model: str,
    diarization_token: str,
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    diarization_pipeline = load_pyannote_pipeline(diarization_model, diarization_token, device)
    diarization_result = diarization_pipeline(
        str(file_path),
        **build_diarization_options(min_speakers=min_speakers, max_speakers=max_speakers),
    )
    annotation = (
        getattr(diarization_result, "exclusive_speaker_diarization", None)
        or getattr(diarization_result, "speaker_diarization", None)
        or diarization_result
    )
    speaker_spans = diarization_annotation_to_spans(annotation)
    enriched_result = clone_result_with_segments(result)
    enriched_result["segments"] = assign_speakers_to_segments(enriched_result.get("segments") or [], speaker_spans)
    return enriched_result


def group_segments_by_speaker(segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for segment in segments:
        if grouped and grouped[-1]["speaker"] == segment.get("speaker"):
            grouped[-1]["end"] = segment["end"]
            grouped[-1]["text"] = normalize_transcript_text(f"{grouped[-1]['text']} {segment['text']}")
            continue
        grouped.append(dict(segment))
    return grouped


def render_plain_segments(
    segments: Sequence[dict[str, Any]],
    include_timestamps: bool,
    cleanup_mode: str,
    speaker_separation: bool,
) -> str:
    if not segments:
        return ""

    has_speakers = any(segment.get("speaker") for segment in segments)
    if speaker_separation and has_speakers:
        lines = []
        for segment in group_segments_by_speaker(segments):
            text = maybe_cleanup_text(str(segment["text"]), cleanup_mode)
            if not text:
                continue

            prefix_parts = []
            if include_timestamps:
                prefix_parts.append(
                    f"[{format_timestamp(float(segment['start']))} -> {format_timestamp(float(segment['end']))}]"
                )
            if segment.get("speaker"):
                prefix_parts.append(f"{segment['speaker']}:")
            prefix = " ".join(prefix_parts)
            lines.append(f"{prefix} {text}".strip())
        return "\n".join(lines)

    if include_timestamps:
        lines = []
        for segment in segments:
            text = maybe_cleanup_text(str(segment["text"]), cleanup_mode)
            if not text:
                continue
            start = format_timestamp(float(segment["start"]))
            end = format_timestamp(float(segment["end"]))
            lines.append(f"[{start} -> {end}] {text}")
        return "\n".join(lines)

    return maybe_cleanup_text(" ".join(str(segment["text"]) for segment in segments), cleanup_mode)


_ACTION_KEYWORDS = (
    "a faire",
    "\xe0 faire",
    "a lancer",
    "\xe0 lancer",
    "je prends",
    "je dois",
    "tu prends",
    "on prend",
    "on doit",
    "il faut",
    "vous devez",
    "vous allez",
    "on va devoir",
    "action",
    "todo",
    "prochaine etape",
    "prochaine \xe9tape",
    "je m'occupe",
    "on s'occupe",
)

_DECISION_KEYWORDS = (
    "on decide",
    "on d\xe9cide",
    "on acte",
    "c'est acte",
    "c'est act\xe9",
    "on valide",
    "c'est valide",
    "c'est valid\xe9",
    "decision",
    "d\xe9cision",
    "accord",
    "arrete",
    "arr\xeat\xe9",
)
_QUESTION_KEYWORDS = (
    "?",
    "a confirmer",
    "reste a voir",
    "reste \xe0 voir",
    "a trancher",
    "\xe0 trancher",
    "on ne sait pas",
    "pas decide",
    "pas d\xe9cid\xe9",
)
_DUE_DATE_PATTERNS = (
    re.compile(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b"),
    re.compile(
        r"\b(\d{1,2}\s+"
        r"(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)"
        r"(?:\s+\d{2,4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(aujourd'hui|demain|apres-demain|apr[eè]s-demain)\b", re.IGNORECASE),
    re.compile(r"\b(semaine prochaine|mois prochain|fin de semaine|fin du mois)\b", re.IGNORECASE),
    re.compile(r"\b(d'ici [^,.;:!?]+)\b", re.IGNORECASE),
    re.compile(r"\b(avant [^,.;:!?]+)\b", re.IGNORECASE),
    re.compile(
        r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)(?: prochain)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(fin T[1-4]|debut T[1-4]|début T[1-4])\b", re.IGNORECASE),
)
ACTION_OWNER_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){0,2})\s+"
    r"(prend|envoie|enverra|doit|va|prepare|prépare|lance|relance|partage|transmet|finalise|finalisé|"
    r"s'occupe|gere|gère|pilote|porte|coordonne)\b"
)
OWNER_STOPWORDS = {
    "Bonjour",
    "Bonsoir",
    "Merci",
    "On",
    "Je",
    "Tu",
    "Vous",
    "Nous",
    "Il",
    "Elle",
    "Ils",
    "Elles",
    "Le",
    "La",
    "Les",
    "Un",
    "Une",
    "Ce",
    "Cette",
    "Ces",
    "Ca",
    "Ça",
}


def extract_action_items(segments: Sequence[dict[str, Any]]) -> list[str]:
    items: list[str] = []
    for segment in segments:
        text = cleanup_transcript_text(str(segment["text"]))
        if len(text) < 15 or not is_action_candidate(text):
            continue
        if text not in items:
            items.append(text)
        if len(items) >= 10:
            break
    return items


def extract_decisions(segments: Sequence[dict[str, Any]]) -> list[str]:
    decisions: list[str] = []
    for segment in segments:
        text = cleanup_transcript_text(str(segment["text"]))
        lowered = text.lower()
        if len(text) < 15 or not any(kw in lowered for kw in _DECISION_KEYWORDS):
            continue
        if text not in decisions:
            decisions.append(text)
        if len(decisions) >= 10:
            break
    return decisions


def extract_open_questions(segments: Sequence[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for segment in segments:
        text = cleanup_transcript_text(str(segment["text"]))
        lowered = text.lower()
        if len(text) < 12 or not any(keyword in lowered for keyword in _QUESTION_KEYWORDS):
            continue
        if text not in questions:
            questions.append(text)
        if len(questions) >= 10:
            break
    return questions


def extract_topic_candidates(segments: Sequence[dict[str, Any]]) -> list[str]:
    topics: list[str] = []
    rejected = set(extract_action_items(segments) + extract_decisions(segments) + extract_open_questions(segments))
    for segment in group_segments_by_speaker(segments):
        text = cleanup_transcript_text(str(segment["text"]))
        lowered = text.lower()
        if len(text) < 18 or text in rejected:
            continue
        if any(keyword in lowered for keyword in _ACTION_KEYWORDS + _DECISION_KEYWORDS):
            continue
        if text not in topics:
            topics.append(text)
        if len(topics) >= 8:
            break
    return topics


def extract_named_owner(text: str) -> str | None:
    match = ACTION_OWNER_RE.search(text)
    if match is None:
        return None

    candidate = cleanup_transcript_text(match.group(1))
    if not candidate or candidate in OWNER_STOPWORDS:
        return None
    return candidate


def is_action_candidate(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _ACTION_KEYWORDS) or ACTION_OWNER_RE.search(text) is not None


def infer_action_owner(text: str, speaker: str | None) -> str:
    named_owner = extract_named_owner(text)
    if named_owner is not None:
        return named_owner

    lowered = text.lower()
    if "je " in lowered or "j'" in lowered:
        return speaker or "Intervenant a confirmer"
    if "on " in lowered:
        return "Collectif"
    if "tu " in lowered or "vous " in lowered:
        return "Destinataire a confirmer"
    return speaker or "Intervenant a confirmer"


def infer_due_date(text: str) -> str:
    for pattern in _DUE_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return cleanup_transcript_text(match.group(1))
    return "Non precisee"


def infer_action_status(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("fait", "termine", "terminé", "boucle", "bouclé", "finalise", "finalisé")):
        return "fait"
    if any(token in lowered for token in ("en cours", "on avance", "on continue", "je continue")):
        return "en cours"
    if any(
        token in lowered
        for token in (
            "a lancer",
            "à lancer",
            "a faire",
            "à faire",
            "a prevoir",
            "à prevoir",
            "a prévoir",
            "reste a faire",
            "reste à faire",
            "todo",
        )
    ):
        return "a faire"
    if any(token in lowered for token in ("on valide", "c'est valide", "c'est validé")):
        return "valide"
    return "a clarifier"


def extract_structured_action_items(segments: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    structured_items: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for segment in segments:
        raw_text = str(segment["text"])
        text = cleanup_transcript_text(raw_text)
        if len(text) < 15 or not is_action_candidate(text):
            continue

        task = text
        if task in seen_tasks:
            continue
        seen_tasks.add(task)

        structured_items.append(
            {
                "owner": infer_action_owner(text, segment.get("speaker")),
                "task": task,
                "due_date": infer_due_date(raw_text),
                "status": infer_action_status(raw_text),
            }
        )
        if len(structured_items) >= 10:
            break
    return structured_items


def shorten_text(text: str, max_length: int) -> str:
    compact = normalize_transcript_text(text)
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def fallback_summary(segments: Sequence[dict[str, Any]], topics: Sequence[str]) -> str:
    if topics:
        return shorten_text("Reunion centree sur " + "; ".join(topics[:3]), 420)

    grouped_segments = group_segments_by_speaker(segments)
    if grouped_segments:
        seed = " ".join(cleanup_transcript_text(str(segment["text"])) for segment in grouped_segments[:3])
        return shorten_text(seed, 420)

    return "Resume automatique indisponible."


def build_meeting_report_fallback(
    segments: Sequence[dict[str, Any]],
    speakers: Sequence[str],
) -> dict[str, Any]:
    topics = extract_topic_candidates(segments)
    decisions = extract_decisions(segments)
    actions = extract_action_items(segments)
    action_items = extract_structured_action_items(segments)
    open_questions = extract_open_questions(segments)
    participants = list(speakers) if speakers else ["Intervenants non identifies"]
    return {
        "participants": participants,
        "summary": fallback_summary(segments, topics),
        "topics": topics or ["Sujet principal non deduit automatiquement."],
        "decisions": decisions or ["Aucune decision detectee automatiquement."],
        "actions": actions or ["Aucune action detectee automatiquement."],
        "action_items": action_items
        or [
            {
                "owner": "A confirmer",
                "task": "Aucune action detectee automatiquement.",
                "due_date": "Non precisee",
                "status": "a clarifier",
            }
        ],
        "open_questions": open_questions or ["Aucune question ouverte detectee automatiquement."],
        "generation_mode": "heuristique locale",
    }


def find_openai_api_key(explicit_key: str | None) -> str | None:
    if explicit_key:
        return explicit_key.strip() or None

    for env_name in OPENAI_API_KEY_ENV_VARS:
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()

    return None


def normalize_string_list(value: Any, fallback_message: str) -> list[str]:
    if isinstance(value, str):
        items = [cleanup_transcript_text(part) for part in value.split("\n")]
    elif isinstance(value, list):
        items = [cleanup_transcript_text(str(part)) for part in value]
    else:
        items = []

    normalized = [item for item in items if item]
    return normalized or [fallback_message]


def normalize_action_items(value: Any, fallback_items: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return [dict(item) for item in fallback_items]

    items: list[dict[str, str]] = []
    for raw_item in value:
        if not isinstance(raw_item, dict):
            continue
        task = cleanup_transcript_text(str(raw_item.get("task", "")))
        if not task:
            continue
        owner = cleanup_transcript_text(str(raw_item.get("owner", ""))) or "A confirmer"
        due_date = cleanup_transcript_text(str(raw_item.get("due_date", ""))) or "Non precisee"
        status = cleanup_transcript_text(str(raw_item.get("status", ""))) or "a clarifier"
        items.append({"owner": owner, "task": task, "due_date": due_date, "status": status})

    return items or [dict(item) for item in fallback_items]


def normalize_meeting_report_payload(payload: Any, fallback_report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Le LLM n'a pas renvoye un objet JSON exploitable.")

    summary = cleanup_transcript_text(str(payload.get("summary", "")))
    report = {
        "participants": normalize_string_list(payload.get("participants"), fallback_report["participants"][0]),
        "summary": summary or fallback_report["summary"],
        "topics": normalize_string_list(payload.get("topics"), fallback_report["topics"][0]),
        "decisions": normalize_string_list(payload.get("decisions"), fallback_report["decisions"][0]),
        "actions": normalize_string_list(payload.get("actions"), fallback_report["actions"][0]),
        "action_items": normalize_action_items(payload.get("action_items"), fallback_report["action_items"]),
        "open_questions": normalize_string_list(payload.get("open_questions"), fallback_report["open_questions"][0]),
        "generation_mode": payload.get("generation_mode") or fallback_report["generation_mode"],
    }
    return report


def request_openai_meeting_report(
    transcript_body: str,
    source_name: str,
    language: str,
    speakers: Sequence[str],
    llm_model: str,
    openai_api_key: str,
) -> dict[str, Any]:
    system_prompt = (
        "Tu rediges des comptes-rendus de reunion en francais a partir d'un verbatim.\n"
        "Reponds en JSON strict avec exactement les cles: "
        "participants, summary, topics, decisions, actions, action_items, open_questions, generation_mode.\n"
        "participants/topics/decisions/actions/open_questions doivent etre des tableaux de chaines.\n"
        "action_items doit etre un tableau d'objets avec exactement les cles owner, task, due_date, status.\n"
        "summary et generation_mode doivent etre des chaines.\n"
        "Ne fabrique pas de noms propres absents du verbatim. Sois concret et concis."
    )
    user_prompt = (
        f"Source: {source_name}\n"
        f"Langue: {language or 'non precisee'}\n"
        f"Intervenants detectes: {', '.join(speakers) if speakers else 'aucun'}\n\n"
        "Verbatim nettoye:\n"
        f"{transcript_body}"
    )
    payload = {
        "model": llm_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erreur OpenAI HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Erreur reseau OpenAI: {exc.reason}") from exc

    data = json.loads(raw_response)
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def maybe_generate_llm_meeting_report(
    fallback_report: dict[str, Any],
    transcript_body: str,
    source_name: str,
    language: str,
    speakers: Sequence[str],
    llm_provider: str,
    llm_model: str,
    openai_api_key: str | None,
) -> dict[str, Any]:
    cache_key = (
        llm_provider,
        llm_model,
        source_name,
        language,
        tuple(speakers),
        transcript_body,
    )
    cached_report = MEETING_REPORT_CACHE.get(cache_key)
    if cached_report is not None:
        return dict(cached_report)

    if llm_provider != "openai":
        MEETING_REPORT_CACHE[cache_key] = dict(fallback_report)
        return dict(fallback_report)

    api_key = find_openai_api_key(openai_api_key)
    if api_key is None:
        console.print("[yellow]Aucune cle OpenAI disponible, fallback vers le CR heuristique local.[/yellow]")
        MEETING_REPORT_CACHE[cache_key] = dict(fallback_report)
        return dict(fallback_report)

    try:
        payload = request_openai_meeting_report(
            transcript_body=transcript_body,
            source_name=source_name,
            language=language,
            speakers=speakers,
            llm_model=llm_model,
            openai_api_key=api_key,
        )
        report = normalize_meeting_report_payload(payload, fallback_report)
        report["generation_mode"] = f"openai:{llm_model}"
        MEETING_REPORT_CACHE[cache_key] = dict(report)
        return report
    except Exception as exc:
        console.print(f"[yellow]Generation LLM indisponible, fallback heuristique utilise:[/yellow] {exc}")
        MEETING_REPORT_CACHE[cache_key] = dict(fallback_report)
        return dict(fallback_report)


def render_bullet_section(title: str, items: Sequence[str]) -> list[str]:
    lines = [title, ""]
    for item in items:
        lines.append(f"- {item}")
    return lines


def render_action_item_line(item: dict[str, str]) -> str:
    return (
        f"- Responsable: {item['owner']} | Tache: {item['task']} | Echeance: {item['due_date']} | Statut: {item['status']}"
    )


def render_meeting_plus_markdown(
    *,
    title: str,
    source_name: str,
    duration_seconds: float | None,
    language: str,
    speakers: Sequence[str],
    report: dict[str, Any],
    verbatim_lines: Sequence[str],
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- Source: `{source_name}`")
    if duration_seconds:
        lines.append(f"- Duree: {format_duration(duration_seconds)}")
    if language:
        lines.append(f"- Langue detectee: `{language}`")
    if speakers:
        lines.append(f"- Intervenants detectes: {', '.join(speakers)}")
    lines.append(f"- Generation CR: `{report['generation_mode']}`")
    lines.extend(["", "## Participants", ""])
    for participant in report["participants"]:
        lines.append(f"- {participant}")
    lines.extend(["", "## Resume", "", report["summary"]])
    lines.extend(["", "## Sujets abordes", ""])
    for item in report["topics"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Decisions", ""])
    for item in report["decisions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Actions", ""])
    for item in report["actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Actions structurees", ""])
    for item in report["action_items"]:
        lines.append(render_action_item_line(item))
    lines.extend(["", "## Questions ouvertes", ""])
    for item in report["open_questions"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Verbatim annexe", ""])
    for line in verbatim_lines:
        lines.append(f"- {line}")
    return "\n".join(lines).strip()


def render_meeting_plus_text(
    *,
    title: str,
    source_name: str,
    duration_seconds: float | None,
    language: str,
    speakers: Sequence[str],
    report: dict[str, Any],
    verbatim_body: str,
) -> str:
    lines = [title, "=" * len(title), "", f"Source: {source_name}"]
    if duration_seconds:
        lines.append(f"Duree: {format_duration(duration_seconds)}")
    if language:
        lines.append(f"Langue detectee: {language}")
    if speakers:
        lines.append(f"Intervenants detectes: {', '.join(speakers)}")
    lines.append(f"Generation CR: {report['generation_mode']}")
    lines.extend(["", "Participants", "------------"])
    for participant in report["participants"]:
        lines.append(f"- {participant}")
    lines.extend(["", "Resume", "------", report["summary"]])
    lines.extend(["", "Sujets abordes", "--------------"])
    for item in report["topics"]:
        lines.append(f"- {item}")
    lines.extend(["", "Decisions", "---------"])
    for item in report["decisions"]:
        lines.append(f"- {item}")
    lines.extend(["", "Actions", "-------"])
    for item in report["actions"]:
        lines.append(f"- {item}")
    lines.extend(["", "Actions structurees", "------------------"])
    for item in report["action_items"]:
        lines.append(render_action_item_line(item))
    lines.extend(["", "Questions ouvertes", "------------------"])
    for item in report["open_questions"]:
        lines.append(f"- {item}")
    lines.extend(["", "Verbatim annexe", "---------------", verbatim_body])
    return "\n".join(lines).strip()


def render_markdown_output(
    result: dict[str, Any],
    include_timestamps: bool,
    cleanup_mode: str,
    speaker_separation: bool,
    title: str,
    source_name: str,
    duration_seconds: float | None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    openai_api_key: str | None = None,
) -> str:
    segments = extract_segments(result)
    speakers = collect_detected_speakers(segments)
    language = normalize_transcript_text(str(result.get("language", "")))
    lines = [f"# {title}", ""]
    lines.append(f"- Source: `{source_name}`")
    if duration_seconds:
        lines.append(f"- Duree: {format_duration(duration_seconds)}")
    if language:
        lines.append(f"- Langue detectee: `{language}`")
    if speakers:
        lines.append(f"- Intervenants detectes: {', '.join(speakers)}")

    if cleanup_mode == "meeting":
        lines.extend(["", "## Points reperes automatiquement", "", "### Actions", ""])
        action_items = extract_action_items(segments)
        if action_items:
            for item in action_items:
                lines.append(f"- {item}")
        else:
            lines.append("- Aucune action detectee automatiquement.")

        lines.extend(["", "### Decisions", ""])
        decisions = extract_decisions(segments)
        if decisions:
            for decision in decisions:
                lines.append(f"- {decision}")
        else:
            lines.append("- Aucune decision detectee automatiquement.")

        lines.extend(["", "## Deroule", ""])
        plain_lines = render_plain_segments(
            segments=segments,
            include_timestamps=include_timestamps or speaker_separation,
            cleanup_mode="clean",
            speaker_separation=speaker_separation,
        ).splitlines()
        for line in plain_lines:
            lines.append(f"- {line}")
        return "\n".join(lines).strip()

    if cleanup_mode == "meeting-plus":
        verbatim_body = render_plain_segments(
            segments=segments,
            include_timestamps=include_timestamps or speaker_separation,
            cleanup_mode="clean",
            speaker_separation=speaker_separation,
        )
        fallback_report = build_meeting_report_fallback(segments, speakers)
        report = maybe_generate_llm_meeting_report(
            fallback_report=fallback_report,
            transcript_body=verbatim_body,
            source_name=source_name,
            language=language,
            speakers=speakers,
            llm_provider=llm_provider,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
        )
        return render_meeting_plus_markdown(
            title=title,
            source_name=source_name,
            duration_seconds=duration_seconds,
            language=language,
            speakers=speakers,
            report=report,
            verbatim_lines=verbatim_body.splitlines() or [verbatim_body],
        )

    lines.extend(["", "## Verbatim", ""])
    body = render_plain_segments(
        segments=segments,
        include_timestamps=include_timestamps,
        cleanup_mode=cleanup_mode,
        speaker_separation=speaker_separation,
    )
    if include_timestamps or (speaker_separation and speakers):
        for line in body.splitlines():
            lines.append(f"- {line}")
    else:
        lines.append(body)
    return "\n".join(lines).strip()


def render_transcript_text(
    result: dict[str, Any],
    include_timestamps: bool,
    cleanup_mode: str = "raw",
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    speaker_separation: bool = False,
    title: str | None = None,
    source_name: str | None = None,
    duration_seconds: float | None = None,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    openai_api_key: str | None = None,
) -> str:
    source_label = source_name or "transcription"
    document_title = title or (
        f"CR enrichi - {source_label}"
        if cleanup_mode == "meeting-plus"
        else f"CR de reunion - {source_label}"
        if cleanup_mode == "meeting"
        else f"Transcription - {source_label}"
    )

    if output_format == "md":
        return render_markdown_output(
            result=result,
            include_timestamps=include_timestamps,
            cleanup_mode=cleanup_mode,
            speaker_separation=speaker_separation,
            title=document_title,
            source_name=source_label,
            duration_seconds=duration_seconds,
            llm_provider=llm_provider,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
        )

    segments = extract_segments(result)
    lines = []
    if cleanup_mode == "meeting":
        lines.append(document_title)
        lines.append("=" * len(document_title))
        lines.append("")
        lines.append(f"Source: {source_label}")
        if duration_seconds:
            lines.append(f"Duree: {format_duration(duration_seconds)}")
        speakers = collect_detected_speakers(segments)
        if speakers:
            lines.append(f"Intervenants detectes: {', '.join(speakers)}")
        lines.append("")
        lines.append("Actions")
        lines.append("-------")
        action_items = extract_action_items(segments)
        if action_items:
            for item in action_items:
                lines.append(f"- {item}")
        else:
            lines.append("- Aucune action detectee automatiquement.")
        lines.append("")
        lines.append("Decisions")
        lines.append("---------")
        decisions = extract_decisions(segments)
        if decisions:
            for decision in decisions:
                lines.append(f"- {decision}")
        else:
            lines.append("- Aucune decision detectee automatiquement.")
        lines.append("")
        lines.append("Deroule")
        lines.append("-------")
        lines.append(
            render_plain_segments(
                segments=segments,
                include_timestamps=include_timestamps or speaker_separation,
                cleanup_mode="clean",
                speaker_separation=speaker_separation,
            )
        )
        return "\n".join(line for line in lines if line is not None).strip()

    if cleanup_mode == "meeting-plus":
        speakers = collect_detected_speakers(segments)
        language = normalize_transcript_text(str(result.get("language", "")))
        verbatim_body = render_plain_segments(
            segments=segments,
            include_timestamps=include_timestamps or speaker_separation,
            cleanup_mode="clean",
            speaker_separation=speaker_separation,
        )
        fallback_report = build_meeting_report_fallback(segments, speakers)
        report = maybe_generate_llm_meeting_report(
            fallback_report=fallback_report,
            transcript_body=verbatim_body,
            source_name=source_label,
            language=language,
            speakers=speakers,
            llm_provider=llm_provider,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
        )
        return render_meeting_plus_text(
            title=document_title,
            source_name=source_label,
            duration_seconds=duration_seconds,
            language=language,
            speakers=speakers,
            report=report,
            verbatim_body=verbatim_body,
        )

    return render_plain_segments(
        segments=segments,
        include_timestamps=include_timestamps,
        cleanup_mode=cleanup_mode,
        speaker_separation=speaker_separation,
    )


def resolve_output_targets(
    input_path: Path,
    output_value: str | None,
    combined_name: str,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> tuple[Path, Path]:
    suffix = OUTPUT_SUFFIXES[output_format]
    if input_path.is_file():
        if output_value:
            output_path = Path(output_value).expanduser()
            if is_supported_output_path(output_path):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                return output_path, output_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path / f"{input_path.stem}{suffix}", output_path

        output_dir = input_path.parent / "transcripts"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_path.stem}{suffix}", output_dir

    if output_value:
        requested_output = Path(output_value).expanduser()
        if is_supported_output_path(requested_output):
            requested_output.parent.mkdir(parents=True, exist_ok=True)
            transcript_dir = requested_output.parent / requested_output.stem
            transcript_dir.mkdir(parents=True, exist_ok=True)
            return requested_output, transcript_dir
        output_root = requested_output
    else:
        output_root = input_path / "transcripts"

    output_root.mkdir(parents=True, exist_ok=True)
    transcript_dir = output_root / "files"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    return output_root / normalize_combined_name(combined_name, output_format), transcript_dir


def build_batch_output_file(
    input_root: Path,
    transcript_dir: Path,
    file_path: Path,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Path:
    relative_path = file_path.relative_to(input_root)
    return transcript_dir / relative_path.with_suffix(OUTPUT_SUFFIXES[output_format])


def relative_source_label(input_root: Path, file_path: Path) -> str:
    return str(file_path.relative_to(input_root))


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip()


def write_text_file(output_file: Path, text: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text.rstrip() + "\n", encoding="utf-8")


def should_skip_existing_output(path: Path, overwrite: bool, skip_existing: bool) -> bool:
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"La sortie attendue est un dossier alors qu'un fichier texte est requis: {path}")

    if not path.exists():
        return False

    if overwrite:
        return False

    if skip_existing:
        return True

    raise FileExistsError(f"La sortie existe deja: {path}. Utilise --overwrite ou --skip-existing.")


def output_will_be_reused(path: Path, overwrite: bool, skip_existing: bool) -> bool:
    return path.exists() and not overwrite and skip_existing


def validate_output_conflicts(
    input_path: Path,
    files: Sequence[Path],
    transcript_dir: Path,
    target_output: Path,
    overwrite: bool,
    skip_existing: bool,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> None:
    should_skip_existing_output(target_output, overwrite, skip_existing)

    if input_path.is_file():
        return

    for file_path in files:
        output_file = build_batch_output_file(input_path, transcript_dir, file_path, output_format=output_format)
        should_skip_existing_output(output_file, overwrite, skip_existing)


def count_pending_transcriptions(
    input_path: Path,
    files: Sequence[Path],
    transcript_dir: Path,
    target_output: Path,
    overwrite: bool,
    skip_existing: bool,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> int:
    if input_path.is_file():
        return 0 if output_will_be_reused(target_output, overwrite, skip_existing) else 1

    pending = 0
    for file_path in files:
        output_file = build_batch_output_file(input_path, transcript_dir, file_path, output_format=output_format)
        if not output_will_be_reused(output_file, overwrite, skip_existing):
            pending += 1
    return pending


def detect_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def build_transcribe_options(
    language: str | None,
    prompt: str | None,
    temperature: float,
    device: str,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "task": "transcribe",
        "verbose": False,
        "temperature": temperature,
    }
    if language:
        options["language"] = language
    if prompt:
        options["initial_prompt"] = prompt
    if device != "cuda":
        options["fp16"] = False
    return options


def run_whisper(
    model: Any,
    file_path: Path,
    language: str | None,
    prompt: str | None,
    temperature: float,
    device: str,
) -> dict[str, Any]:
    if model is None:
        raise RuntimeError("Aucun modele Whisper n'est charge alors qu'une transcription est necessaire.")
    options = build_transcribe_options(language, prompt, temperature, device)
    return model.transcribe(str(file_path), **options)  # type: ignore[no-any-return]


def transcribe_source(
    model: Any,
    file_path: Path,
    language: str | None,
    prompt: str | None,
    temperature: float,
    device: str,
    diarize: bool,
    diarization_model: str,
    diarization_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    result = run_whisper(model, file_path, language, prompt, temperature, device)
    if not diarize:
        return result

    token = find_huggingface_token(diarization_token)
    if token is None:
        raise RuntimeError(
            "La diarisation requiert un token Hugging Face. "
            f"Passe `--hf-token` ou definis une variable parmi: {', '.join(DEFAULT_HF_TOKEN_ENV_VARS)}."
        )

    return apply_speaker_diarization(
        result=result,
        file_path=file_path,
        diarization_model=diarization_model,
        diarization_token=token,
        device=device,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def render_transcription(
    result: dict[str, Any],
    file_path: Path,
    include_timestamps: bool,
    cleanup_mode: str,
    output_format: str,
    speaker_separation: bool,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    llm_model: str = DEFAULT_LLM_MODEL,
    openai_api_key: str | None = None,
) -> str:
    text = render_transcript_text(
        result=result,
        include_timestamps=include_timestamps,
        cleanup_mode=cleanup_mode,
        output_format=output_format,
        speaker_separation=speaker_separation,
        source_name=file_path.name,
        duration_seconds=probe_duration_seconds(file_path),
        llm_provider=llm_provider,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
    )
    if not text:
        raise RuntimeError(f"Aucun texte n'a ete produit pour {file_path.name}")
    return text


def transcribe_media_file(
    model: Any,
    file_path: Path,
    language: str | None,
    prompt: str | None,
    temperature: float,
    include_timestamps: bool,
    cleanup_mode: str,
    output_format: str,
    speaker_separation: bool,
    device: str,
    diarize: bool,
    diarization_model: str,
    diarization_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    llm_provider: str,
    llm_model: str,
    openai_api_key: str | None,
) -> str:
    result = transcribe_source(
        model=model,
        file_path=file_path,
        language=language,
        prompt=prompt,
        temperature=temperature,
        device=device,
        diarize=diarize,
        diarization_model=diarization_model,
        diarization_token=diarization_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return render_transcription(
        result,
        file_path,
        include_timestamps,
        cleanup_mode,
        output_format,
        speaker_separation,
        llm_provider=llm_provider,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
    )


def load_whisper_model(whisper_module: Any, model_name: str, device: str) -> tuple[Any, str]:
    try:
        return whisper_module.load_model(model_name, device=device), device
    except TypeError:
        return whisper_module.load_model(model_name), "cpu"
    except Exception:
        if device == "cpu":
            raise

        console.print(f"[yellow]Echec du chargement sur {device}, repli sur cpu.[/yellow]")
        return whisper_module.load_model(model_name, device="cpu"), "cpu"


def transcribe_to_output(
    model: Any,
    source_file: Path,
    output_file: Path,
    language: str | None,
    prompt: str | None,
    temperature: float,
    include_timestamps: bool,
    cleanup_mode: str,
    output_format: str,
    speaker_separation: bool,
    overwrite: bool,
    skip_existing: bool,
    device: str,
    diarize: bool,
    diarization_model: str,
    diarization_token: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    llm_provider: str,
    llm_model: str,
    openai_api_key: str | None,
) -> bool:
    if output_format == "both":
        txt_file = output_file.with_suffix(".txt")
        md_file = output_file.with_suffix(".md")
        txt_skip = should_skip_existing_output(txt_file, overwrite, skip_existing)
        md_skip = should_skip_existing_output(md_file, overwrite, skip_existing)
        if txt_skip and md_skip:
            return True
        result = transcribe_source(
            model=model,
            file_path=source_file,
            language=language,
            prompt=prompt,
            temperature=temperature,
            device=device,
            diarize=diarize,
            diarization_model=diarization_model,
            diarization_token=diarization_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if not txt_skip:
            txt_text = render_transcription(
                result,
                source_file,
                include_timestamps,
                cleanup_mode,
                "txt",
                speaker_separation,
                llm_provider=llm_provider,
                llm_model=llm_model,
                openai_api_key=openai_api_key,
            )
            write_text_file(txt_file, txt_text)
        if not md_skip:
            md_text = render_transcription(
                result,
                source_file,
                include_timestamps,
                cleanup_mode,
                "md",
                speaker_separation,
                llm_provider=llm_provider,
                llm_model=llm_model,
                openai_api_key=openai_api_key,
            )
            write_text_file(md_file, md_text)
        return False

    if should_skip_existing_output(output_file, overwrite, skip_existing):
        return True

    text = transcribe_media_file(
        model=model,
        file_path=source_file,
        language=language,
        prompt=prompt,
        temperature=temperature,
        include_timestamps=include_timestamps,
        cleanup_mode=cleanup_mode,
        output_format=output_format,
        speaker_separation=speaker_separation,
        device=device,
        diarize=diarize,
        diarization_model=diarization_model,
        diarization_token=diarization_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        llm_provider=llm_provider,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
    )
    write_text_file(output_file, text)
    return False


def write_combined_transcript(
    combined_output: Path,
    transcript_index: Sequence[tuple[str, Path]],
    overwrite: bool,
    skip_existing: bool,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> bool:
    if should_skip_existing_output(combined_output, overwrite, skip_existing):
        console.print(f"[yellow]Transcription complete deja presente, fichier conserve:[/yellow] {combined_output}")
        return False

    combined_output.parent.mkdir(parents=True, exist_ok=True)
    with combined_output.open("w", encoding="utf-8") as handle:
        if output_format == "md":
            handle.write("# Transcriptions combinees\n")
        for index, (label, transcript_file) in enumerate(transcript_index):
            if index:
                handle.write("\n\n")
            if output_format == "md":
                handle.write(f"\n## {label}\n\n")
            else:
                handle.write(f"--- {label} ---\n")
            handle.write(read_text_file(transcript_file))
            handle.write("\n")
    return True


def transcribe_single_file(model: Any, input_file: Path, output_file: Path, args: argparse.Namespace, device: str) -> int:
    reused = transcribe_to_output(
        model=model,
        source_file=input_file,
        output_file=output_file,
        language=args.langue,
        prompt=args.prompt,
        temperature=args.temperature,
        include_timestamps=args.timestamps,
        cleanup_mode=args.mode_rendu,
        output_format=args.format,
        speaker_separation=args.speaker_separation,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        device=device,
        diarize=args.diarize,
        diarization_model=args.diarization_model,
        diarization_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        openai_api_key=args.openai_api_key,
    )

    if reused:
        console.print(f"[yellow]Transcription existante reutilisee:[/yellow] {output_file}")
    else:
        console.print(f"[green]Transcription ecrite dans[/green] {output_file}")

    return 0


def transcribe_directory(
    model: Any,
    input_root: Path,
    files: Sequence[Path],
    transcript_dir: Path,
    combined_output: Path,
    args: argparse.Namespace,
    device: str,
) -> int:
    transcript_index: list[tuple[str, Path]] = []
    processed_count = 0
    reused_count = 0
    errors: list[tuple[Path, str]] = []
    skip_combined = output_will_be_reused(combined_output, args.overwrite, args.skip_existing)

    if Progress is not None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            TextColumn(" | [cyan]{task.fields[current]}[/cyan]"),
            console=console,  # type: ignore[arg-type]
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[bold yellow]Transcription des fichiers...[/bold yellow]",
                total=len(files),
                current="",
            )

            for file_path in files:
                label = relative_source_label(input_root, file_path)
                progress.update(task, current=label)
                output_file = build_batch_output_file(
                    input_root,
                    transcript_dir,
                    file_path,
                    output_format=args.format,
                )

                try:
                    reused = transcribe_to_output(
                        model=model,
                        source_file=file_path,
                        output_file=output_file,
                        language=args.langue,
                        prompt=args.prompt,
                        temperature=args.temperature,
                        include_timestamps=args.timestamps,
                        cleanup_mode=args.mode_rendu,
                        output_format=args.format,
                        speaker_separation=args.speaker_separation,
                        overwrite=args.overwrite,
                        skip_existing=args.skip_existing,
                        device=device,
                        diarize=args.diarize,
                        diarization_model=args.diarization_model,
                        diarization_token=args.hf_token,
                        min_speakers=args.min_speakers,
                        max_speakers=args.max_speakers,
                        llm_provider=args.llm_provider,
                        llm_model=args.llm_model,
                        openai_api_key=args.openai_api_key,
                    )
                    transcript_index.append((label, output_file))
                    if reused:
                        reused_count += 1
                    else:
                        processed_count += 1
                except Exception as exc:
                    errors.append((file_path, str(exc)))
                    if not args.continue_on_error:
                        raise
                finally:
                    progress.update(task, advance=1)
    else:
        for index, file_path in enumerate(files, start=1):
            label = relative_source_label(input_root, file_path)
            console.print(f"[cyan]{index}/{len(files)}[/cyan] {label}")
            output_file = build_batch_output_file(
                input_root,
                transcript_dir,
                file_path,
                output_format=args.format,
            )

            try:
                reused = transcribe_to_output(
                    model=model,
                    source_file=file_path,
                    output_file=output_file,
                    language=args.langue,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    include_timestamps=args.timestamps,
                    cleanup_mode=args.mode_rendu,
                    output_format=args.format,
                    speaker_separation=args.speaker_separation,
                    overwrite=args.overwrite,
                    skip_existing=args.skip_existing,
                    device=device,
                    diarize=args.diarize,
                    diarization_model=args.diarization_model,
                    diarization_token=args.hf_token,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    openai_api_key=args.openai_api_key,
                )
                transcript_index.append((label, output_file))
                if reused:
                    reused_count += 1
                else:
                    processed_count += 1
            except Exception as exc:
                errors.append((file_path, str(exc)))
                if not args.continue_on_error:
                    raise

    if not transcript_index:
        raise RuntimeError("Aucune transcription n'a pu etre produite.")

    if skip_combined:
        console.print(f"[yellow]Transcription complete deja presente, fichier conserve:[/yellow] {combined_output}")
    else:
        write_combined_transcript(
            combined_output=combined_output,
            transcript_index=transcript_index,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            output_format=args.format,
        )

    console.print(f"[green]Transcriptions individuelles dans[/green] {transcript_dir}")
    console.print(f"[green]Transcription complete dans[/green] {combined_output}")
    console.print(
        f"[blue]Bilan:[/blue] {processed_count} nouveau(x) fichier(s), {reused_count} reutilise(s), {len(errors)} erreur(s)."
    )

    if errors:
        for file_path, message in errors[:5]:
            console.print(f"[red]- {relative_source_label(input_root, file_path)}:[/red] {message}")
        if len(errors) > 5:
            console.print(f"[red]... {len(errors) - 5} erreur(s) supplementaire(s).[/red]")
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcrit un fichier ou un dossier de fichiers audio/video en texte exploitable."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Chemin vers un fichier audio/video ou un dossier contenant les fichiers a transcrire.",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        required=False,
        help="Chemin vers un fichier audio/video ou un dossier contenant les fichiers a transcrire.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Fichier .txt/.md cible pour un fichier unique, ou dossier racine de sortie pour un lot.",
    )
    parser.add_argument(
        "--langue",
        "--language",
        dest="langue",
        default=None,
        help="Langue explicite (ex: fr, en, French). Laisse vide pour detection automatique.",
    )
    parser.add_argument(
        "--modele",
        "--model",
        dest="modele",
        default=DEFAULT_MODEL,
        help="Modele Whisper. `large` pour privilegier la precision, `turbo` pour aller plus vite.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device PyTorch pour Whisper. `auto` essaie cuda, puis mps, puis cpu.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Contexte initial pour aider Whisper sur les noms propres, acronymes ou jargon metier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature de decoding Whisper. 0.0 est le plus stable pour un verbatim exploitable.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Ajoute les timestamps par segment dans le fichier de sortie.",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_OUTPUT_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help="Format de sortie: `txt` pour un verbatim simple, `md` pour un rendu exploitable dans un doc.",
    )
    parser.add_argument(
        "--mode-rendu",
        "--render-mode",
        dest="mode_rendu",
        choices=("raw", "clean", "meeting", "meeting-plus"),
        default="raw",
        help="`raw` pour le verbatim brut, `clean` pour nettoyer le verbatim, `meeting` pour un CR structure, `meeting-plus` pour un CR enrichi.",
    )
    parser.add_argument(
        "--speaker-separation",
        action="store_true",
        help="Separe les interventions si Whisper ou un backend annexe fournit des labels speaker.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Active une diarisation optionnelle via pyannote.audio pour tenter d'identifier les intervenants.",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help="Modele Hugging Face de diarisation a charger avec pyannote.audio.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Token Hugging Face pour la diarisation. Par securite, prefere une variable d'environnement.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Nombre minimum d'intervenants attendu pour aider la diarisation.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Nombre maximum d'intervenants attendu pour aider la diarisation.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=("none", "openai"),
        default=DEFAULT_LLM_PROVIDER,
        help="Generation optionnelle du CR enrichi via LLM. `none` ne consomme aucune API externe.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Modele OpenAI utilise si `--llm-provider openai` est active.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Cle OpenAI explicite. Par securite, prefere `OPENAI_API_KEY`.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourt les sous-dossiers quand l'entree est un dossier.",
    )
    parser.add_argument(
        "--combined-name",
        default=None,
        help="Nom du fichier combine pour un traitement par dossier. L'extension suit `--format`.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reutilise les .txt deja presents et saute leur retranscription.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ecrase les fichiers de sortie existants.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue les autres fichiers meme si l'un d'eux echoue.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.input = args.input_option or args.input_path
    args.combined_name = normalize_combined_name(
        args.combined_name or default_combined_filename(args.format),
        args.format,
    )

    if args.skip_existing and args.overwrite:
        parser.error("--skip-existing et --overwrite sont incompatibles.")

    if args.temperature < 0:
        parser.error("--temperature doit etre positive ou nulle.")

    if args.min_speakers is not None and args.min_speakers <= 0:
        parser.error("--min-speakers doit etre strictement positif.")

    if args.max_speakers is not None and args.max_speakers <= 0:
        parser.error("--max-speakers doit etre strictement positif.")

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers ne peut pas etre superieur a --max-speakers.")

    if args.output:
        output_path = Path(args.output).expanduser()
        if is_supported_output_path(output_path) and output_path.suffix.lower() != OUTPUT_SUFFIXES[args.format]:
            parser.error(
                f"--output utilise l'extension `{output_path.suffix}`, mais --format `{args.format}` attend "
                f"`{OUTPUT_SUFFIXES[args.format]}`."
            )

    if args.diarize:
        args.speaker_separation = True

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input:
        args.input = input("Chemin du fichier ou dossier a transcrire: ").strip()

    if not args.input:
        console.print("[bold red]Erreur:[/bold red] aucun chemin fourni.")
        return 1

    input_path = Path(args.input).expanduser()

    try:
        ensure_ffmpeg_available()
        whisper = load_whisper_module()
        files = collect_input_files(input_path, recursive=args.recursive)
        target_output, transcript_dir = resolve_output_targets(
            input_path,
            args.output,
            args.combined_name,
            output_format=args.format,
        )
        validate_output_conflicts(
            input_path=input_path,
            files=files,
            transcript_dir=transcript_dir,
            target_output=target_output,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
            output_format=args.format,
        )
    except Exception as exc:
        console.print(f"[bold red]Erreur:[/bold red] {exc}")
        return 1

    total_duration = estimate_total_duration_seconds(files)
    if total_duration:
        console.print(f"[blue]Duree totale detectee:[/blue] {format_duration(total_duration)}")

    pending_transcriptions = count_pending_transcriptions(
        input_path=input_path,
        files=files,
        transcript_dir=transcript_dir,
        target_output=target_output,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        output_format=args.format,
    )

    device = detect_device(args.device)
    model = None
    active_device = device

    if pending_transcriptions:
        console.print(f"[bold green]Chargement du modele Whisper:[/bold green] {args.modele} ({device})")
        try:
            model, active_device = load_whisper_model(whisper, args.modele, device)
        except Exception as exc:
            console.print(f"[bold red]Impossible de charger le modele `{args.modele}`:[/bold red] {exc}")
            return 1
    else:
        console.print("[blue]Aucune nouvelle transcription necessaire, reutilisation des fichiers existants.[/blue]")

    try:
        if input_path.is_file():
            return transcribe_single_file(model, files[0], target_output, args, active_device)
        return transcribe_directory(model, input_path, files, transcript_dir, target_output, args, active_device)
    except KeyboardInterrupt:
        console.print("\n[bold red]Transcription interrompue.[/bold red]")
        return 130
    except Exception as exc:
        console.print(f"[bold red]Echec pendant la transcription:[/bold red] {exc}")
        return 1
