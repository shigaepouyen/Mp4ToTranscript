from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mp4_to_transcript.cli import (
    assign_speakers_to_segments,
    build_batch_output_file,
    build_temperature_schedule,
    build_transcribe_options,
    clean_macos_malloc_environment,
    cleanup_transcript_text,
    collect_input_files,
    count_pending_transcriptions,
    extract_structured_action_items,
    format_timestamp,
    parse_args,
    render_transcript_text,
    resolve_output_targets,
)


class CollectInputFilesTests(unittest.TestCase):
    def test_collect_input_files_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "meeting.m4a"
            source.write_text("placeholder", encoding="utf-8")

            files = collect_input_files(source)

            self.assertEqual(files, [source])

    def test_collect_input_files_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "nested").mkdir()
            top_level = root / "a.mp3"
            nested = root / "nested" / "b.wav"
            ignored = root / "notes.txt"
            top_level.write_text("placeholder", encoding="utf-8")
            nested.write_text("placeholder", encoding="utf-8")
            ignored.write_text("placeholder", encoding="utf-8")

            files = collect_input_files(root, recursive=True)

            self.assertEqual(files, [top_level, nested])


class OutputResolutionTests(unittest.TestCase):
    def test_resolve_output_targets_for_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "meeting.m4a"
            source.write_text("placeholder", encoding="utf-8")

            target_output, transcript_dir = resolve_output_targets(source, None, "bundle.txt")

            self.assertEqual(target_output, Path(tmp_dir) / "transcripts" / "meeting.txt")
            self.assertEqual(transcript_dir, Path(tmp_dir) / "transcripts")

    def test_resolve_output_targets_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir) / "audios"
            source_dir.mkdir()

            target_output, transcript_dir = resolve_output_targets(source_dir, None, "mon_lot.txt")

            self.assertEqual(target_output, source_dir / "transcripts" / "mon_lot.txt")
            self.assertEqual(transcript_dir, source_dir / "transcripts" / "files")

    def test_build_batch_output_file_preserves_relative_path(self) -> None:
        input_root = Path("/tmp/input")
        transcript_dir = Path("/tmp/output/files")
        file_path = input_root / "clients" / "meeting.mp4"

        output_file = build_batch_output_file(input_root, transcript_dir, file_path)

        self.assertEqual(output_file, transcript_dir / "clients" / "meeting.txt")

    def test_count_pending_transcriptions_uses_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "audios"
            source_dir.mkdir()
            media_file = source_dir / "meeting.mp3"
            media_file.write_text("placeholder", encoding="utf-8")
            combined_output, transcript_dir = resolve_output_targets(source_dir, None, "bundle.txt")
            transcript_file = build_batch_output_file(source_dir, transcript_dir, media_file)
            transcript_file.parent.mkdir(parents=True, exist_ok=True)
            transcript_file.write_text("already done", encoding="utf-8")

            pending = count_pending_transcriptions(
                input_path=source_dir,
                files=[media_file],
                transcript_dir=transcript_dir,
                target_output=combined_output,
                overwrite=False,
                skip_existing=True,
            )

            self.assertEqual(pending, 0)


class TranscriptRenderingTests(unittest.TestCase):
    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(3661.7), "01:01:01")

    def test_render_transcript_with_timestamps(self) -> None:
        result = {
            "text": "Texte global",
            "segments": [
                {"start": 0.0, "end": 4.4, "text": " Bonjour  a tous "},
                {"start": 4.4, "end": 8.2, "text": " on commence "},
            ],
        }

        rendered = render_transcript_text(result, include_timestamps=True)

        self.assertEqual(
            rendered,
            "[00:00:00 -> 00:00:04] Bonjour a tous\n[00:00:04 -> 00:00:08] on commence",
        )

    def test_cleanup_transcript_text_removes_simple_disfluencies(self) -> None:
        self.assertEqual(cleanup_transcript_text("euh bonjour , on avance"), "Bonjour, on avance")

    def test_render_transcript_markdown_meeting_mode(self) -> None:
        result = {
            "language": "fr",
            "segments": [
                {"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00", "text": "bonjour a tous"},
                {"start": 6.0, "end": 14.0, "speaker": "SPEAKER_01", "text": "on valide le budget et je prends l'action"},
            ],
        }

        rendered = render_transcript_text(
            result,
            include_timestamps=False,
            cleanup_mode="meeting",
            output_format="md",
            speaker_separation=True,
            source_name="reunion.m4a",
            duration_seconds=14.0,
        )

        self.assertIn("# CR de reunion - reunion.m4a", rendered)
        self.assertIn("- Intervenants detectes: Intervenant 1, Intervenant 2", rendered)
        self.assertIn("## Points reperes automatiquement", rendered)
        self.assertIn("## Deroule", rendered)

    def test_render_transcript_with_speaker_separation(self) -> None:
        result = {
            "segments": [
                {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "bonjour"},
                {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_00", "text": "on commence"},
                {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_01", "text": "tres bien"},
            ],
        }

        rendered = render_transcript_text(
            result,
            include_timestamps=True,
            speaker_separation=True,
        )

        self.assertEqual(
            rendered,
            "[00:00:00 -> 00:00:08] Intervenant 1: bonjour on commence\n"
            "[00:00:08 -> 00:00:12] Intervenant 2: tres bien",
        )

    def test_assign_speakers_to_segments_uses_strongest_overlap(self) -> None:
        segments = [
            {"start": 0.0, "end": 3.0, "text": "bonjour"},
            {"start": 3.0, "end": 7.0, "text": "on avance"},
        ]
        speaker_spans = [
            {"start": 0.0, "end": 2.0, "speaker": "Intervenant 1"},
            {"start": 2.0, "end": 6.5, "speaker": "Intervenant 2"},
        ]

        assigned = assign_speakers_to_segments(segments, speaker_spans)

        self.assertEqual(assigned[0]["speaker"], "Intervenant 1")
        self.assertEqual(assigned[1]["speaker"], "Intervenant 2")

    def test_extract_structured_action_items_infers_owner_and_due_date(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 6.0,
                "speaker": "Intervenant 2",
                "text": "je prends le suivi du budget d'ici vendredi",
            }
        ]

        items = extract_structured_action_items(segments)

        self.assertEqual(
            items,
            [
                {
                    "owner": "Intervenant 2",
                    "task": "Je prends le suivi du budget d'ici vendredi",
                    "due_date": "D'ici vendredi",
                    "status": "a clarifier",
                }
            ],
        )

    def test_extract_structured_action_items_detects_richer_due_date_and_status(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 7.0,
                "speaker": "Intervenant 1",
                "text": "on doit lancer la communication avant le prochain CA, a lancer",
            }
        ]

        items = extract_structured_action_items(segments)

        self.assertEqual(
            items,
            [
                {
                    "owner": "Collectif",
                    "task": "On doit lancer la communication avant le prochain CA, a lancer",
                    "due_date": "Avant le prochain CA",
                    "status": "a faire",
                }
            ],
        )

    def test_extract_structured_action_items_prefers_named_owner_over_speaker(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 7.0,
                "speaker": "Intervenant 1",
                "text": "Paul prend le suivi budget pour le 15 avril",
            }
        ]

        items = extract_structured_action_items(segments)

        self.assertEqual(
            items,
            [
                {
                    "owner": "Paul",
                    "task": "Paul prend le suivi budget pour le 15 avril",
                    "due_date": "15 avril",
                    "status": "a clarifier",
                }
            ],
        )

    def test_render_transcript_markdown_meeting_plus_mode(self) -> None:
        result = {
            "language": "fr",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "bonjour on fait le point sur le budget"},
                {"start": 5.0, "end": 11.0, "speaker": "SPEAKER_01", "text": "Marie envoie le compte-rendu"},
                {"start": 11.0, "end": 16.0, "speaker": "SPEAKER_00", "text": "reste a voir la date de lancement ?"},
            ],
        }

        rendered = render_transcript_text(
            result,
            include_timestamps=False,
            cleanup_mode="meeting-plus",
            output_format="md",
            speaker_separation=True,
            source_name="reunion.m4a",
            duration_seconds=16.0,
        )

        self.assertIn("# CR enrichi - reunion.m4a", rendered)
        self.assertIn("## Participants", rendered)
        self.assertIn("## Resume", rendered)
        self.assertIn("## Sujets abordes", rendered)
        self.assertIn("## Actions structurees", rendered)
        self.assertIn("Responsable: Marie", rendered)
        self.assertIn("Statut:", rendered)
        self.assertIn("## Verbatim annexe", rendered)
        self.assertIn("- Generation CR: `heuristique locale`", rendered)


class WhisperOptionsTests(unittest.TestCase):
    def test_clean_macos_malloc_environment_removes_noisy_native_flags(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"MallocStackLogging": "0", "MallocStackLoggingNoCompact": "0", "OTHER": "kept"},
            clear=True,
        ):
            cleaned = clean_macos_malloc_environment()

            self.assertNotIn("MallocStackLogging", cleaned)
            self.assertNotIn("MallocStackLoggingNoCompact", cleaned)
            self.assertEqual(cleaned["OTHER"], "kept")

    def test_temperature_schedule_matches_whisper_fallback_defaults(self) -> None:
        self.assertEqual(
            build_temperature_schedule(0.0, 0.2),
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )

    def test_transcribe_options_avoid_previous_text_repetition_loops_by_default(self) -> None:
        options = build_transcribe_options(language="fr", prompt="Contexte", temperature=0.0, device="cpu")

        self.assertFalse(options["condition_on_previous_text"])
        self.assertEqual(options["temperature"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertFalse(options["fp16"])

    def test_hallucination_silence_threshold_enables_word_timestamps(self) -> None:
        options = build_transcribe_options(
            language="fr",
            prompt=None,
            temperature=0.0,
            device="cpu",
            hallucination_silence_threshold=2.0,
        )

        self.assertTrue(options["word_timestamps"])
        self.assertEqual(options["hallucination_silence_threshold"], 2.0)


class OutputFormatTests(unittest.TestCase):
    def test_resolve_output_targets_for_single_file_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "meeting.m4a"
            source.write_text("placeholder", encoding="utf-8")

            target_output, transcript_dir = resolve_output_targets(source, None, "bundle", output_format="md")

            self.assertEqual(target_output, Path(tmp_dir) / "transcripts" / "meeting.md")
            self.assertEqual(transcript_dir, Path(tmp_dir) / "transcripts")

    def test_build_batch_output_file_uses_markdown_extension(self) -> None:
        input_root = Path("/tmp/input")
        transcript_dir = Path("/tmp/output/files")
        file_path = input_root / "clients" / "meeting.mp4"

        output_file = build_batch_output_file(input_root, transcript_dir, file_path, output_format="md")

        self.assertEqual(output_file, transcript_dir / "clients" / "meeting.md")


class ArgParsingTests(unittest.TestCase):
    def test_parse_args_enables_speaker_separation_when_diarize_is_set(self) -> None:
        args = parse_args(["--input", "/tmp/meeting.m4a", "--diarize"])

        self.assertTrue(args.diarize)
        self.assertTrue(args.speaker_separation)

    def test_parse_args_keeps_explicit_hf_token(self) -> None:
        args = parse_args(["--input", "/tmp/meeting.m4a", "--diarize", "--hf-token", "hf_test_token"])

        self.assertEqual(args.hf_token, "hf_test_token")
        self.assertTrue(args.diarize)

    def test_parse_args_supports_llm_options(self) -> None:
        args = parse_args(
            [
                "--input",
                "/tmp/meeting.m4a",
                "--mode-rendu",
                "meeting-plus",
                "--llm-provider",
                "openai",
                "--llm-model",
                "gpt-5-mini",
                "--openai-api-key",
                "sk-test",
            ]
        )

        self.assertEqual(args.mode_rendu, "meeting-plus")
        self.assertEqual(args.llm_provider, "openai")
        self.assertEqual(args.llm_model, "gpt-5-mini")
        self.assertEqual(args.openai_api_key, "sk-test")

    def test_parse_args_enables_word_timestamps_for_hallucination_guard(self) -> None:
        args = parse_args(
            [
                "--input",
                "/tmp/meeting.m4a",
                "--hallucination-silence-threshold",
                "2.0",
            ]
        )

        self.assertTrue(args.word_timestamps)
        self.assertEqual(args.hallucination_silence_threshold, 2.0)

    def test_parse_args_can_reenable_previous_text_context(self) -> None:
        args = parse_args(["--input", "/tmp/meeting.m4a", "--condition-on-previous-text"])

        self.assertTrue(args.condition_on_previous_text)


if __name__ == "__main__":
    unittest.main()
