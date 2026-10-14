# Mp4ToTranscript

> **Requiert Apple Silicon (M1 ou superieur).** Ce projet utilise `mlx-whisper`, qui exploite le GPU Metal via le framework MLX d'Apple. Il ne fonctionne pas sur x86/Intel ni sur Linux/Windows.

`Mp4ToTranscript` est un outil en ligne de commande qui transcrit des fichiers audio ou video avec Whisper (via MLX) et produit des fichiers texte (`.txt`) ou Markdown (`.md`).

Il peut traiter un fichier unique ou un dossier complet, ajouter des timestamps, nettoyer le verbatim, produire une structure de compte-rendu, tenter une separation par intervenant et, en option, enrichir le compte-rendu avec OpenAI.

## Application Mac

Une interface locale est disponible en complément de la commande existante :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[desktop]"
.venv/bin/python -m mp4_to_transcript.desktop
```

Pour créer l’application autonome utilisable par double-clic ou depuis le Dock
(FFmpeg doit être installé sur le Mac de construction) :

```bash
.venv/bin/python -m pip install -e ".[desktop,build]"
.venv/bin/python scripts/build_mac_app.py
open dist/Mp4ToTranscript.app
```

**Où est l’application ?** Le script crée `Mp4ToTranscript.app` dans le dossier
`dist` du projet. Dans le Finder, ouvrir ce dossier puis double-cliquer sur
`Mp4ToTranscript.app`. Il n’est pas nécessaire de relancer les commandes
d’installation à chaque utilisation.

Pour un accès quotidien, déplacer `Mp4ToTranscript.app` dans Applications puis
la glisser dans le Dock. Depuis la version 0.5, le `.app` embarque Python, Qt,
MLX/Whisper, FFmpeg et FFprobe : il peut être déplacé sans le projet ni `.venv`.
L’ensemble du `.app` doit rester intact. Les modèles téléchargés et le cache de
transcription sont conservés séparément sur le Mac et réutilisés après déplacement.

Le glisser-déposer est pris en charge **dans la fenêtre et sur l’icône du Dock**,
même si l’app est fermée. Les fichiers ou dossiers sont ajoutés à la file ;
choisir ensuite un profil et cliquer sur Transcrire. Un dépôt ne déclenche pas
automatiquement de traitement ni d’appel OpenAI. « Ouvrir avec » dans le Finder
fonctionne également. L’app ne remplace pas votre lecteur multimédia par défaut.
L’application dispose d’une icône dans le Finder et le Dock. Après mise à jour,
relancer l’application pour voir la nouvelle icône. Sa source vectorielle est
dans `mp4_to_transcript/assets/app-icon.svg` ; `scripts/build_icon.py` régénère
les fichiers PNG et ICNS.

L’app cible Apple Silicon. Elle contient une signature locale de construction,
mais pas une signature Developer ID ni une notarisation Apple pour une diffusion
publique. Les modèles ne sont pas inclus : un modèle absent nécessite un premier
téléchargement. La diarisation reste une fonction de la CLI, pas de l’interface.

La version Qt est fixée à 6.8.3, validée avec une vraie fenêtre macOS. Le journal
du lanceur se trouve dans `~/Library/Logs/Mp4ToTranscript/launch.log`.
Le build autonome utilise PyInstaller et conserve la réception des fichiers
envoyés par macOS. Pour les développeurs, `scripts/build_mac_app.py --linked`
produit encore un petit lanceur lié au Python du projet ; **cette variante seule**
nécessite de conserver `.venv` et le projet à leur emplacement. Elle utilise
`xcrun clang` et un Python macOS de type framework.

Si le Finder conserve une ancienne icône après reconstruction, quitter puis
relancer l’app. Si nécessaire, retirer uniquement son raccourci du Dock et y
glisser la nouvelle copie du `.app`.

### Utilisation

1. Déposer des fichiers ou dossiers **dans la fenêtre**, ou utiliser les boutons
   Ajouter. Les sous-dossiers sont inclus et les doublons ignorés.
2. Choisir Texte nettoyé, Transcription brute ou Compte rendu, puis Transcrire.
3. Sélectionner un résultat pour le prévisualiser, le copier ou l’afficher dans le Finder.

**Lire en grand** ouvre le transcript complet dans une fenêtre indépendante,
redimensionnable, avec plein écran, zoom et copie. Échap quitte le plein écran.

Une description s’affiche sous le profil sélectionné :

- **Texte nettoyé** : conserve le contenu, réduit certaines hésitations et
  répétitions, harmonise la ponctuation ; ne résume pas.
- **Transcription brute** : texte reconnu par Whisper, avec un minimum de retouches.
- **Compte rendu** : organise le contenu en résumé, sujets, décisions et actions,
  avec le verbatim en annexe. Utilise des règles locales ; la synthèse OpenAI
  nécessite d’activer explicitement cette option.

Le traitement affiche cinq étapes : vérification du fichier, vérification du
modèle, préparation du moteur, transcription, création du résultat. Le temps
écoulé reste visible pendant les étapes longues. Il ne s’agit pas d’un pourcentage
estimé de transcription.

L’app vérifie la configuration et les poids du modèle dans le cache local avant
toute requête de téléchargement. S’ils sont présents, elle transmet leur chemin
local à Whisper et indique « aucun téléchargement du modèle ». Si ces fichiers
manquent, elle annonce le téléchargement. Charger un modèle **en mémoire** n’est
pas le télécharger. Une transcription déjà en cache évite même ce chargement.

Les réglages (langue, format, modèle, timestamps, contexte et destination) sont
mémorisés. Les fichiers sont traités successivement dans un processus séparé.
Arrêter la file annule le traitement courant et conserve les suivants en attente.
« Remettre en attente » permet de réessayer ou de créer un autre rendu.
La file elle-même est conservée uniquement pendant la session.

Les exports ne remplacent jamais un fichier existant : un suffixe numérique est
ajouté en cas de collision. Par défaut, ils sont dans `transcripts`, à côté de
chaque source. Un dossier ajouté produit des exports individuels ; le regroupement
et la diarisation restent disponibles dans la CLI.

Le cache local de transcription se trouve dans `~/Library/Caches/Mp4ToTranscript`
(modifiable avec `MP4_TRANSCRIPT_CACHE_DIR`). Changer le format, le profil ou les
timestamps réutilise ce cache. Changer le contenu audio, la langue, le modèle ou
le contexte recalcule la transcription. Ce cache contient du texte et peut être
supprimé à tout moment lorsque l’app est arrêtée.

OpenAI est désactivé au lancement. L’option d’enrichissement, dans Réglages,
envoie **le texte** à OpenAI et utilise une clé saisie pour la session ou
`OPENAI_API_KEY`. La clé n’est pas enregistrée. Sans cette option, le compte rendu
est structuré par les règles locales existantes, pas par un modèle de synthèse.
Le moteur existant revient au rendu local si OpenAI échoue ; le mode de génération
est indiqué dans le compte rendu. Le premier usage d’un modèle absent du Mac
nécessite son téléchargement depuis Hugging Face.

### Validation

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
```

Test macOS de l’ouverture via le Dock/Finder (app isolée, fichiers de test,
fenêtre refermée automatiquement) :

```bash
.venv/bin/python scripts/test_mac_open.py
```

Pour vérifier la version autonome construite :

```bash
.venv/bin/python scripts/test_mac_open.py --app dist/Mp4ToTranscript.app
.venv/bin/python scripts/test_standalone.py dist/Mp4ToTranscript.app /chemin/vers/test.wav
```

Le second test déplace une copie de l’app dans un dossier temporaire, contrôle
ses dépendances natives, ouvre la fenêtre et effectue une transcription avec
un cache de transcription vide et les téléchargements désactivés. Le modèle
`mlx-community/whisper-large-v3-turbo` doit déjà être présent sur le Mac
(ou préciser `--model`). Aucun appel OpenAI n’est effectué.

## Prerequis

- Apple Silicon (M1, M2, M3, M4 ou superieur)
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

- `--modele`, `--model`: repo HF du modele mlx-whisper, par exemple `mlx-community/whisper-large-v3-mlx` ou `mlx-community/whisper-medium-mlx`.
- `--device`: ignore avec mlx-whisper, Metal est utilise automatiquement.
- `--langue`, `--language`: langue de l'audio, par exemple `fr`.
- `--prompt`: contexte donne a Whisper.
- `--temperature`: temperature de depart du decoding.
- `--temperature-increment-on-fallback`: increment utilise lorsque Whisper retente un segment.
- `--condition-on-previous-text` / `--no-condition-on-previous-text`: reutilisation du texte precedent comme contexte.
- `--carry-initial-prompt` / `--no-carry-initial-prompt`: sans effet avec mlx-whisper (option conservee pour compatibilite).
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
