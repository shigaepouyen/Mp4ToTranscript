# Mp4ToTranscript

Petit projet CLI pour prendre un fichier audio/video en entree et produire un fichier texte exploitable (`.txt` ou `.md`) avec Whisper.

L'objectif reste volontairement simple:

- entree: un fichier audio/video ou un dossier de fichiers
- sortie: un ou plusieurs fichiers `.txt` ou `.md`
- usage cible: verbatims exploitables ensuite pour des CR, comptes-rendus, notes ou documentation

Depuis cette version, le rendu peut aussi sortir en `.md`, proposer un verbatim nettoye, generer un premier "CR de reunion" structure, sortir un `meeting-plus` plus exploitable et tenter une separation par intervenant via diarisation optionnelle.

## Ce que j'ai ameliore par rapport au script initial

- projet range dans un dossier dedie
- packaging minimal avec `pyproject.toml`
- wrapper compatible `Mp4ToTranscript.py`
- README d'installation et d'usage
- gestion plus sure des sorties: plus d'ecrasement silencieux
- options `--skip-existing` et `--overwrite`
- traitement batch plus propre avec un fichier combine et des fichiers individuels ranges dans `files/`
- preservation de l'arborescence en mode dossier, ce qui evite les collisions de noms
- option `--recursive` pour parcourir des sous-dossiers
- option `--timestamps` pour garder des reperes temporels
- option `--format md` pour un export Markdown exploitable tel quel
- option `--mode-rendu clean` pour nettoyer automatiquement le verbatim
- option `--mode-rendu meeting` pour generer un CR de reunion structure
- option `--mode-rendu meeting-plus` pour un CR enrichi avec resume, sujets, decisions, actions, actions structurees et questions ouvertes
- actions structurees avec extraction best effort de `responsable / tache / echeance / statut`
- option `--speaker-separation` pour separer les interventions quand des labels locuteur sont disponibles
- option `--diarize` pour produire ces labels avec `pyannote.audio` quand l'audio le permet
- option `--llm-provider openai` pour enrichir le `meeting-plus` via OpenAI de facon optionnelle
- detection automatique du device (`cuda`, `mps`, `cpu`) avec repli sur CPU
- mode "reprise" possible si des transcriptions existent deja
- quelques tests unitaires sur les fonctions critiques hors modele

## Audit rapide du code initial

Le script d'origine etait deja utile, mais presentait plusieurs fragilites pour un usage regulier:

1. les sorties existantes etaient ecrasees sans confirmation
2. en batch, les fichiers de sortie individuels pouvaient entrer en collision si plusieurs sources avaient le meme nom
3. il n'y avait ni packaging, ni README, ni tests
4. le traitement par dossier n'etait pas concu pour une reprise propre
5. la structure restait monolithique, donc plus difficile a maintenir

## Prerequis

- Python 3.10+
- FFmpeg installe sur la machine
- dependances Python installees

### Installer FFmpeg sur macOS

```bash
brew install ffmpeg
```

## Installation

Depuis le dossier du projet:

```bash
cd Mp4ToTranscript
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Optionnel, installation editable:

```bash
python3 -m pip install -e .
```

Optionnel, pour la separation par intervenant:

```bash
python3 -m pip install -e ".[diarization]"
```

Puis definir un token Hugging Face:

```bash
export HF_TOKEN="hf_xxx"
```

## Utilisation

### 1. Fichier unique vers un `.txt`

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a
```

Sortie par defaut:

```text
/chemin/vers/transcripts/reunion.txt
```

### 2. Fichier unique avec sortie explicite

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a --output /chemin/vers/reunion.txt
```

### 3. Dossier complet

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/audios
```

Sorties par defaut:

```text
/chemin/vers/audios/transcripts/transcription_complete.txt
/chemin/vers/audios/transcripts/files/*.txt
```

### 4. Dossier complet en recursive

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/audios --recursive
```

Dans ce cas, la structure des sous-dossiers est conservee dans `files/`.

### 5. Ajouter les timestamps

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/reunion.m4a --timestamps
```

### 6. Exporter en Markdown nettoye

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu clean
```

### 7. Generer un premier CR de reunion

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting \
  --speaker-separation
```

### 8. Generer un CR enrichi local

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting-plus
```

### 9. Generer un CR enrichi avec OpenAI

```bash
export OPENAI_API_KEY="sk-xxx"

python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting-plus \
  --llm-provider openai \
  --llm-model gpt-5-mini
```

Si l'appel OpenAI echoue, le script revient automatiquement au mode heuristique local.

### 9.b Exemple de sortie `meeting-plus`

```text
## Actions structurees

- Responsable: Paul | Tache: Paul prend le suivi budget pour le 15 avril | Echeance: 15 avril | Statut: a clarifier
- Responsable: Collectif | Tache: On doit lancer la communication avant le prochain CA, a lancer | Echeance: Avant le prochain CA | Statut: a faire
```

### 10. Tenter une separation par intervenant

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --format md \
  --mode-rendu meeting \
  --diarize \
  --min-speakers 2 \
  --max-speakers 6
```

`--diarize` active automatiquement `--speaker-separation`.

### 11. Reprendre un lot sans retraiter les fichiers deja faits

```bash
python3 Mp4ToTranscript.py --input /chemin/vers/audios --recursive --skip-existing
```

### 12. Aider Whisper avec du jargon metier

```bash
python3 Mp4ToTranscript.py \
  --input /chemin/vers/reunion.m4a \
  --langue fr \
  --prompt "Contexte: reunion projet, budget, planning, clients, livrables."
```

## Options utiles

```bash
python3 Mp4ToTranscript.py --help
```

Options principales:

- `--modele large` pour privilegier la qualite
- `--modele turbo` pour aller plus vite
- `--langue fr` pour eviter une mauvaise detection
- `--timestamps` pour un verbatim source plus facile a citer
- `--format md` pour produire un document Markdown
- `--mode-rendu clean` pour nettoyer les hesitations et la ponctuation
- `--mode-rendu meeting` pour une sortie "CR de reunion"
- `--mode-rendu meeting-plus` pour un CR plus actionnable
- `--speaker-separation` pour exploiter les labels speaker quand ils existent
- `--diarize` pour tenter de generer ces labels avec `pyannote.audio`
- `--hf-token` ou la variable d'environnement `HF_TOKEN` pour autoriser le modele de diarisation
- `--min-speakers` / `--max-speakers` pour cadrer le nombre d'intervenants
- `--llm-provider openai` pour enrichir `meeting-plus` avec un LLM
- `--llm-model` pour choisir le modele OpenAI
- `--openai-api-key` ou `OPENAI_API_KEY` pour autoriser l'appel OpenAI
- `--skip-existing` pour reprendre un lot
- `--overwrite` pour regenarer les sorties
- `--continue-on-error` pour ne pas bloquer tout un batch sur un seul fichier

## Conseils d'usage pour des CR et docs

- Pour un texte propre a retravailler ensuite dans un doc, commence sans `--timestamps`.
- Pour une relecture ou pour retrouver une citation dans l'audio, active `--timestamps`.
- Pour un CR rapidement partageable, essaye `--format md --mode-rendu meeting`.
- Pour un CR plus propre sans editer a la main, essaye `--format md --mode-rendu meeting-plus`.
- `meeting-plus` tente aussi de structurer les actions en `responsable / tache / echeance / statut`.
- Les echeances reconnaissent maintenant mieux des formes comme `15 avril`, `avant le prochain CA`, `fin T2`.
- Les responsables nommes directement dans le verbatim sont aussi reperes dans des formulations comme `Paul prend`, `Marie envoie`, `Jean doit`.
- Avec `--llm-provider openai`, le meme schema est conserve, mais le remplissage des sections et des actions est en general plus propre.
- Pour un verbatim plus lisible sans changer le fond, essaye `--mode-rendu clean`.
- `--speaker-separation` ne fait effet que si des labels locuteur sont presents dans les segments fournis au rendu.
- Si tu veux vraiment identifier les intervenants, ajoute `--diarize` et un token Hugging Face.
- `--llm-provider openai` n'est jamais obligatoire: sans cle API, le script peut rester 100% local.
- Si l'audio contient beaucoup de noms propres ou de jargon, passe un `--prompt`.
- Si la machine rame, essaye `--modele turbo`.

## Structure du projet

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

## Limites connues

- Le mode `meeting` fournit une premiere structure de CR, pas un resume "intelligent" garanti: une relecture humaine reste utile.
- Le mode `meeting-plus` heuristique reste approximatif tant qu'aucun LLM n'est active.
- La qualite depend fortement de l'audio source.
- Le modele `large` est plus lent et plus gourmand en RAM.
- La separation par intervenant reste "best effort": elle depend de la qualite micro, des chevauchements de voix et du nombre d'intervenants.
- La diarisation necessite une dependance optionnelle (`pyannote.audio`) et un token Hugging Face pour charger le modele.
- Le mode OpenAI consomme une API externe seulement si `--llm-provider openai` est demande explicitement.
