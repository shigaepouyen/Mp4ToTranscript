# Prompt pour la session Claude Code locale : migration SwiftUI

À coller dans Claude Code, lancé **sur le Mac** à la racine du dépôt (`claude`), une fois la PR de l'audit fusionnée dans `main`.

---

Tu es un développeur macOS expérimenté (Swift, SwiftUI, AppKit) qui connaît les Human Interface Guidelines d'Apple. Tu travailles sur mon Mac, sous macOS 26, avec Xcode : tu peux compiler, lancer l'app, prendre des captures d'écran et corriger toi-même.

## Contexte

Mp4ToTranscript transforme un fichier audio ou vidéo en transcription ou en compte rendu Markdown, avec Whisper (MLX, Apple Silicon). Lis d'abord :
- `docs/audit/audit-ux-ui.md` : l'audit complet. Le **lot 3** décrit l'architecture cible ; les **sections 2 et 2 bis** listent les défauts à ne pas reproduire ;
- `mp4_to_transcript/desktop_worker.py` : le moteur, piloté en JSON-lines sur stdin et stdout. Il est testé : **c'est l'interface à conserver** ;
- `scripts/Mp4ToTranscript.spec` et `scripts/test_standalone.py` : l'empaquetage PyInstaller actuel, qui embarque Python, MLX et FFmpeg ;
- `mp4_to_transcript/desktop.py` : l'interface Qt actuelle, que tu vas remplacer. Elle sert de référence de comportement, pas de modèle de design.

## Décisions déjà prises (ne pas les rediscuter)

- **Interface SwiftUI native, macOS 26 minimum, arm64 uniquement.** Utiliser les API récentes et Liquid Glass, sans code de compatibilité.
- **Le moteur Python ne change pas.** Il est embarqué comme exécutable d'aide PyInstaller (`--worker`), sans Qt, dans `Contents/Helpers/`. Le protocole JSON-lines et `cli.py` ne bougent pas.
- **Usage : un fichier à la fois.** Le dépôt d'un fichier (dans la fenêtre, sur l'icône du Dock, ou via « Ouvrir avec ») **lance la transcription immédiatement**. Seule exception : si l'enrichissement OpenAI est actif, demander confirmation.
- **Fenêtre unique et compacte**, sans icône dans la barre des menus. Le contenu a trois états : dépôt → progression → résultat. Le résultat s'affiche sur une carte avec Coup d'œil (Espace), Ouvrir et Afficher dans le Finder. **Pas d'aperçu texte intégré ni de lecteur** : je lis mes transcriptions dans un autre outil.
- **Barre d'outils** : menu Profil (Texte nettoyé, Transcription brute, Compte rendu, avec la description de chaque profil dans le menu), menu Langue (Français, Anglais, Automatique), bouton Historique.
- **Historique persistant** dans une barre latérale repliable, masquée par défaut. Chaque entrée affiche le nom, la date, la durée, le profil et l'état. Actions : Coup d'œil, Ouvrir, Finder, Retranscrire, Retirer. Les chemins sont stockés sous forme de signets (bookmarks).
- **Réglages** dans une scène `Settings` (Cmd+,), sans bouton de validation, avec trois sections :
  - Général : format, destination, repères de temps, contexte ;
  - Modèles : modèle par défaut **large-v3**, taille de chaque modèle téléchargé, bouton de suppression ;
  - Enrichissement : OpenAI désactivé par défaut, clé stockée dans le **Trousseau**.
- **Pas de séparation des intervenants dans l'app.** Elle reste disponible en ligne de commande uniquement.
- **Pas de compte Apple Developer** : signature ad hoc (`codesign -s -`) de tout le bundle, sans notarisation. Mes collègues (tous sous macOS 26) lèveront le blocage de Gatekeeper avec « Ouvrir quand même ». Prévoir que le passage en Developer ID se fasse en une seule étape, le jour où je prendrai un compte.
- Menus complets : Fichier, Édition, Présentation, Fenêtre, Aide. Raccourcis : Cmd+O, Cmd+. pour arrêter, Espace pour Coup d'œil, Cmd+, pour les Réglages.
- Notifications natives à la fin d'un traitement, seulement si l'app est en arrière-plan, et cliquables pour ouvrir le résultat. Progression affichée sur l'icône du Dock.
- Interface en français.

## Méthode

Travaille par étapes. **Arrête-toi à la fin de chaque étape** : montre-moi une capture de l'app qui tourne, dis ce qui est fait et ce qui ne l'est pas, et attends mon accord avant de passer à la suite.

1. **Squelette et moteur.** Créer un projet Xcode `macos/` à côté du code Python. Construire le worker PyInstaller sans Qt (un spec dédié, ou une variante de l'actuel). L'intégrer comme exécutable d'aide, puis lancer une transcription réelle depuis une vue SwiftUI minimale : dépôt d'un fichier, réception des événements JSON, affichage du résultat. Valider la signature ad hoc de l'ensemble et le lancement depuis `/Applications`.
2. **Fenêtre principale** : les trois états, la barre d'outils, le lancement au dépôt, le dépôt sur l'icône du Dock, la confirmation OpenAI, l'annulation propre.
3. **Résultat et historique** : la carte de résultat, Coup d'œil, la barre latérale et sa persistance.
4. **Réglages, Trousseau et modèles**, avec la progression du téléchargement en octets.
5. **Menus, raccourcis, notifications, progression dans le Dock, accessibilité** : libellés VoiceOver, mode sombre, Augmenter le contraste.
6. **Icône** dans Icon Composer (format `.icon`), à partir du concept actuel : une page avec une forme d'onde.
7. **Empaquetage** : un script unique qui produit `Mp4ToTranscript.app` signé ad hoc, plus une adaptation de `scripts/test_standalone.py` à ce nouveau bundle.

## Règles

- Ne modifie pas `cli.py` ni le protocole du worker sans me le demander.
- Garde l'app Qt fonctionnelle jusqu'à ce que je valide la version SwiftUI.
- Avant chaque commit, lance les tests Python existants (`python3 -m unittest discover -s tests`), compile l'app Swift et vérifie le fonctionnement dans l'app réelle.
- Distingue toujours ce que tu as vérifié en lançant l'app de ce que tu supposes.
- Écris en français. Sois critique : si une décision ci-dessus pose un problème technique réel, dis-le avant de coder, au lieu de le contourner en silence.
