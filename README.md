# Mp4ToTranscript

`Mp4ToTranscript` est un outil en ligne de commande qui transcrit des fichiers audio ou video avec Whisper et produit des fichiers texte (`.txt`) ou Markdown (`.md`).

Il peut traiter un fichier unique ou un dossier complet, ajouter des timestamps, nettoyer le verbatim, produire une structure de compte-rendu, tenter une separation par intervenant et, en option, enrichir le compte-rendu avec OpenAI.

## Prerequis

- Python 3.10+
- FFmpeg et FFprobe disponibles dans le terminal
- Dependances Python du projet

Installation de FFmpeg sur macOS:

```bash
brew install ffmpeg
```

## Installation

Dans le dossier du projet:

```bash
cd Mp4ToTranscript
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Installation editable optionnelle:

```bash
python3 -m pip install -e .
```

Dependances optionnelles pour la diarisation:

```bash
python3 -m pip install -e ".[diarization]"
```

La diarisation utilise Hugging Face. Definis un token avant d'utiliser `--diarize`:

```bash
export HF_TOKEN="hf_xxx"
```

## Commande De Base

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a
```

Pour afficher toutes les options:

```bash
python3 Mp4ToTranscript.py --help
```

## Entrees Acceptees

L'option `--input` accepte:

- un fichier audio ou video;
- un dossier contenant des fichiers audio ou video;
- un dossier avec sous-dossiers si `--recursive` est utilise.

Extensions supportees:

```text
.aac .flac .m4a .m4v .mkv .mov .mp3 .mp4 .mpeg .mpga .ogg .opus .wav .webm .wma
```

## Sorties

Pour un fichier unique, la sortie par defaut est creee dans un dossier `transcripts` place a cote du fichier source:

```text
/chemin/vers/transcripts/reunion.txt
```

Pour un dossier, deux types de sorties sont crees:

```text
/chemin/vers/audios/transcripts/transcription_complete.txt
/chemin/vers/audios/transcripts/files/*.txt
```

En mode dossier, `transcription_complete.txt` regroupe les transcriptions individuelles. Les fichiers individuels sont ranges dans `files/`. Avec `--recursive`, l'arborescence des sous-dossiers est conservee.

## Exemples D'usage

### Fichier Unique

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a
```

### Fichier Unique Avec Sortie Explicite

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --output /chemin/vers/reunion.txt
```

### Dossier Complet

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/audios
```

### Dossier Recursif

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/audios \
  --recursive
```

### Transcription Avec Timestamps

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --timestamps
```

### Markdown

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md
```

### Verbatim Nettoye

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --mode-rendu clean
```

### Compte-Rendu Structure

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting
```

### Compte-Rendu Enrichi Local

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting-plus
```

### Compte-Rendu Enrichi Avec OpenAI

```bash
export OPENAI_API_KEY="sk-xxx"

python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting-plus \
  --llm-provider openai \
  --llm-model gpt-5-mini
```

### Jargon, Acronymes Et Noms Propres

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --langue fr \
  --prompt "Contexte: reunion projet, budget, planning, clients, livrables."
```

### Reprise D'un Lot

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/audios \
  --recursive \
  --skip-existing
```

### Regeneration D'une Sortie

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --overwrite
```

## Formats De Sortie

`--format txt`

Produit un fichier texte simple.

`--format md`

Produit un fichier Markdown avec titres et sections quand le mode de rendu le permet.

`--format both`

Produit une sortie `.txt` et une sortie `.md` pour chaque source.

## Modes De Rendu

`--mode-rendu raw`

Transcription brute fournie par Whisper, apres normalisation minimale des espaces.

`--mode-rendu clean`

Verbatim nettoye: hesitations simples, espaces et ponctuation sont normalises.

`--mode-rendu meeting`

Compte-rendu structure avec sections de base, actions detectees, decisions detectees et deroule.

`--mode-rendu meeting-plus`

Compte-rendu enrichi avec participants, resume, sujets, decisions, actions, actions structurees, questions ouvertes et verbatim annexe.

## Options Principales

### Source Et Sortie

- `--input`: fichier ou dossier source.
- `--output`: fichier cible pour une source unique, ou dossier racine pour un lot.
- `--recursive`: parcours des sous-dossiers.
- `--combined-name`: nom du fichier combine en mode dossier.
- `--skip-existing`: reutilise les sorties deja presentes.
- `--overwrite`: ecrase les sorties existantes.
- `--continue-on-error`: continue un lot meme si un fichier echoue.

### Whisper

- `--modele`, `--model`: modele Whisper a charger, par exemple `large` ou `turbo`.
- `--device`: `auto`, `cpu`, `cuda` ou `mps`.
- `--langue`, `--language`: langue de l'audio, par exemple `fr`.
- `--prompt`: contexte donne a Whisper.
- `--temperature`: temperature de depart du decoding.
- `--temperature-increment-on-fallback`: increment utilise lorsque Whisper retente un segment.
- `--condition-on-previous-text` / `--no-condition-on-previous-text`: reutilisation du texte precedent comme contexte.
- `--carry-initial-prompt` / `--no-carry-initial-prompt`: repetition du prompt a chaque fenetre Whisper.
- `--compression-ratio-threshold`: seuil de detection des sorties repetitives.
- `--logprob-threshold`: seuil de fiabilite moyenne.
- `--no-speech-threshold`: seuil de detection du silence.
- `--word-timestamps`: timestamps au niveau des mots.
- `--hallucination-silence-threshold`: filtrage de certains segments autour de silences longs.

### Rendu

- `--timestamps`: ajoute les timestamps par segment.
- `--format`: `txt`, `md` ou `both`.
- `--mode-rendu`: `raw`, `clean`, `meeting` ou `meeting-plus`.
- `--speaker-separation`: regroupe les segments par locuteur quand des labels sont disponibles.

### Diarisation

- `--diarize`: active la diarisation avec `pyannote.audio`.
- `--diarization-model`: modele Hugging Face utilise pour la diarisation.
- `--hf-token`: token Hugging Face explicite.
- `--min-speakers`: nombre minimum d'intervenants attendu.
- `--max-speakers`: nombre maximum d'intervenants attendu.

`--diarize` active automatiquement `--speaker-separation`.

### OpenAI

- `--llm-provider openai`: utilise OpenAI pour generer les sections de `meeting-plus`.
- `--llm-model`: modele OpenAI utilise.
- `--openai-api-key`: cle API explicite.

La variable d'environnement `OPENAI_API_KEY` peut aussi etre utilisee.

## Exemple De Sortie `meeting-plus`

```text
# CR enrichi - reunion.m4a

- Source: `reunion.m4a`
- Duree: 47m 21s
- Langue detectee: `fr`
- Generation CR: `heuristique locale`

## Participants

- Intervenants non identifies

## Resume

Resume automatique de la reunion.

## Sujets abordes

- Sujet principal detecte dans le verbatim.

## Decisions

- Decision detectee automatiquement.

## Actions

- Action detectee automatiquement.

## Actions structurees

- Responsable: Paul | Tache: Paul prend le suivi budget pour le 15 avril | Echeance: 15 avril | Statut: a clarifier
```

## Messages Et Depannage

### Sortie Existante

Si un fichier de sortie existe deja, le script s'arrete sauf si une option de reprise est fournie:

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a --skip-existing
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a --overwrite
```

### Warning macOS `MallocStackLogging`

Sur macOS, Python peut afficher ce message sur stderr:

```text
Python(...) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
```

Ce warning provient du runtime macOS/Python. Si le fichier de sortie est produit, le message n'indique pas un echec de transcription.

Une commande avec environnement nettoye peut etre utilisee si le message gene la lecture des logs:

```bash
env -u MallocStackLogging -u MallocStackLoggingNoCompact python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --langue fr \
  --overwrite
```

### Sortie Repetitive

Si Whisper produit une repetition longue sur un audio difficile, ajoute des options de segmentation et de contexte:

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --langue fr \
  --prompt "Contexte: reunion projet, budget, planning, clients, livrables." \
  --carry-initial-prompt \
  --hallucination-silence-threshold 2.0 \
  --overwrite
```

`--hallucination-silence-threshold` active les timestamps par mot et peut augmenter la duree de traitement.

## Structure Du Projet

```text
Mp4ToTranscript/
├── Mp4ToTranscript.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── mp4_to_transcript/
│   ├── __init__.py
│   └── cli.py
└── tests/
    └── test_cli.py
```
