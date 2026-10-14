# Audit UX/UI et technique : Mp4ToTranscript (macOS)

*Audit du 14 octobre 2026, fait sur la branche `claude/mp4-transcript-macos-audit-tdvi96` (commit `eee8db3`).*

> **Aucune capture d'écran n'était disponible** : le dossier `docs/audit/screenshots/` n'existe pas.
> Tous les constats viennent donc de la lecture du code ([Vu dans le code]) ou d'une déduction ([Hypothèse]).
> Aucun constat n'est marqué [Vu sur capture]. La liste des captures à fournir figure en section 5.
>
> L'environnement d'audit tourne sous Linux : l'application n'a été ni compilée ni lancée.

## Cadrage retenu

Ce cadrage s'appuie sur tes réponses aux questions de la phase 1.

| # | Sujet | Décision | Effet sur l'audit |
|---|---|---|---|
| 1 | Pour qui | Usage personnel, mais à donner à 1 ou 2 collègues. **Pas de compte Apple Developer pour l'instant.** Les Mac des collègues sont fournis par l'entreprise, avec le droit de passer outre Gatekeeper | Python et FFmpeg doivent être embarqués. Sans compte, l'app ne peut avoir qu'une signature *ad hoc*, sans notarisation : chaque collègue devra lever le blocage de Gatekeeper à la main (voir T2) |
| 2 | Changer de stack | Oui | La recommandation SwiftUI est chiffrée et assumée (lot 3) |
| 3 | Usage | Un fichier à la fois | La file d'attente permanente n'est pas justifiée à l'écran |
| 4 | Dépôt | Le dépôt lance la transcription | Profil et langue doivent être visibles *avant* le dépôt |
| 5 | Forme | Une fenêtre compacte, plus le dépôt sur l'icône du Dock | Pas d'app de barre des menus ; l'icône de barre des menus actuelle est à retirer |
| 6 | Lecture | Le résultat se lit dans un autre outil ; l'app montre un aperçu Coup d'œil | L'aperçu intégré et le lecteur « Lire en grand » sont à supprimer |
| 7 | Séparation des intervenants | **Non**, pas dans l'app | Elle reste disponible en ligne de commande. Pas de pyannote ni de torch dans l'app, donc pas de jeton Hugging Face à demander |
| 8 | Langue | Français, anglais moins souvent | La langue reste visible, en compact ; le format passe dans les Réglages |
| 9 | OpenAI | Accord sur l'option par défaut | La clé va dans le Trousseau ; OpenAI reste désactivé par défaut |
| 10 | Modèle par défaut | « Le plus performant » | large-v3 reste le défaut, avec une réserve (voir T8) |
| 11 | Historique | Requis | La fenêtre a besoin d'une barre latérale repliable |
| 12 | Matériel | Apple Silicon uniquement | Confirmé |

**Sur la question 7.** La séparation des intervenants (« Intervenant 1 : … ») n'existe qu'en ligne de commande (`--diarize`, `cli.py:329-434`). Tu as confirmé qu'elle ne doit pas entrer dans l'app. Il faudra donc décrire l'app sans cette option, au lieu de la mentionner comme tu l'avais fait au départ.

**Réserve sur la question 10.** « Performant » est ambigu. Si cela veut dire « le plus précis », c'est large-v3, le réglage actuel. Si cela veut dire « le meilleur compromis », c'est probablement large-v3-turbo : une précision proche en français, un traitement plusieurs fois plus rapide, et environ la moitié du poids et de la mémoire [Hypothèse fondée sur les caractéristiques publiques des modèles]. Je garde large-v3 par défaut, conformément à ta réponse. Mais je recommande de comparer les deux sur 2 ou 3 de tes propres enregistrements avant de figer ce choix.

---

## 1. Résumé

**Verdict : un bon moteur, une coque qui n'est pas native.** Le découpage entre l'interface et un processus de travail qui communique en JSON est sain, testé et annulable. Mais l'interface ressemble à un formulaire Qt habillé à la main, pas à une app Mac.

**Les 3 problèmes les plus graves :**
1. **L'apparence native est désactivée volontairement.** Le style Fusion, la palette claire forcée et la feuille de style maison suppriment le mode sombre, la couleur d'accentuation et l'aspect des contrôles système (`desktop.py:550-610`).
2. **La fenêtre est trop grande et trop chargée pour un usage à un seul fichier.** Elle impose un minimum de 820×650, empile 11 bandes, fait doublon entre la zone de dépôt et la file, et affiche trois lignes de statut. Il n'y a ni barre des menus, ni Cmd+, ni historique.
3. **L'app n'est pas transmissible.** Le lanceur dépend du `.venv` du développeur et d'un FFmpeg installé par Homebrew. Sans compte développeur, même une app autonome restera bloquée par Gatekeeper chez tes collègues : il faudra contourner ce blocage à la main.

**Est-ce une vraie app Mac aujourd'hui ?** Non. **Peut-elle le devenir avec la stack actuelle ?** En partie : en Qt, on peut atteindre environ 70 % du résultat (style natif, menus, Cmd+, fenêtre compacte). Mais la barre d'outils unifiée, la fenêtre Réglages au standard macOS, Coup d'œil, les notifications, le Trousseau, l'icône au format macOS 26 et un bundle autonome exigent tous de contourner Qt. **Ma recommandation : une interface SwiftUI qui pilote le worker Python existant**, avec un Python relocalisable embarqué dans l'app. Cela représente environ 13 à 22 jours de travail, sans notarisation.

---

## 2. Tableau des constats

Abréviations de gravité : **B** = Bloquant, **Ma** = Majeur, **Mi** = Mineur. L'effort est noté S, M ou L. Sauf mention contraire, les références de fichier renvoient à `mp4_to_transcript/`.

### Design

| ID | Constat | Statut | Réf. HIG | Grav. | Effort | Recommandation | Fichier:lignes |
|---|---|---|---|---|---|---|---|
| D1 | Le style **Fusion** est forcé et une palette claire est codée en dur. Les contrôles ne sont donc pas ceux d'AppKit, et le mode sombre comme la couleur d'accentuation système sont ignorés. En mode sombre, on aurait probablement une barre de titre sombre au-dessus d'un contenu clair. | Style et palette : [Vu dans le code]. Rendu en mode sombre : [Hypothèse] | Dark Mode ; Color ; Accessibility | **B** | S | Supprimer `setStyle("Fusion")`, la palette et la feuille de style pour revenir au style `macos` natif de Qt. En SwiftUI, c'est le comportement par défaut. | `desktop.py:599-610` |
| D2 | La feuille de style recrée boutons, champs, tableaux et barre de progression (rayons, bordures, survols, vert `#3e5026`). `DESIGN.md` annonce des « contrôles Qt standard » : c'est contradictoire. | [Vu dans le code] | Buttons ; Color (« prefer system colors ») | Ma | S | Supprimer la feuille de style. L'accent vert ne doit vivre que dans l'icône. | `desktop.py:550-569`, `DESIGN.md` |
| D3 | La taille de police est figée à 13 px pour tout le texte, et le titre est à 25 px. Le titre « Vos enregistrements, en texte. » est un slogan : aucune app utilitaire Apple n'affiche de titre marketing dans son contenu. | [Vu dans le code] | Typography ; Writing | Ma | S | Retirer le titre ; le nom de la fenêtre suffit. Utiliser les styles de texte système (`.body`, `.headline`, `.caption`). | `desktop.py:127-129`, `551-553` |
| D4 | La zone de dépôt est un `QLabel` en pointillés : ni icône, ni retour au survol, ni changement d'état. | [Vu dans le code] | Drag and drop (« provide feedback ») | Ma | S | En Qt : surligner la zone pendant `dragEnter`/`dragLeave` et refuser le dépôt pour les formats non pris en charge. En SwiftUI : `.dropDestination` avec `isTargeted`, un SF Symbol `waveform` et la couleur d'accentuation. | `desktop.py:130-134`, `310-316` |
| D5 | **Les icônes système manquent partout.** Les boutons sont textuels (« Ajouter des fichiers… », « A− », « A+ », « Plein écran »). L'icône de barre des menus reprend l'icône Qt générique « volume ». | [Vu dans le code] | SF Symbols ; Toolbars | Mi | S (SwiftUI) | Utiliser des SF Symbols dans une barre d'outils : `plus`, `stop.fill`, `sidebar.left`, `eye` (Coup d'œil). | `desktop.py:40-48`, `136-141`, `279` |
| D6 | L'icône d'app est dessinée à la main, avec une ombre intégrée et une tuile de 896 px sur 1024 (le gabarit Apple est de 824 px). Ce rectangle arrondi n'est pas la forme continue officielle. Sous macOS 26, une icône non conforme risque d'être affichée réduite dans un fond gris générique, et de paraître plus grande que ses voisines sur les versions antérieures. | Géométrie : [Vu dans le code]. Rendu sous macOS 26 : [Hypothèse] | App icons | Ma | M | Refaire l'icône avec **Icon Composer** (format `.icon`, couches pour l'effet Liquid Glass). Garder le concept « page + forme d'onde », qui est lisible. | `assets/app-icon.svg`, `scripts/build_icon.py` |
| D7 | Il y a trois lignes de statut empilées (`status`, `detail`, `model_notice`), plus un chronomètre et une barre de progression de 5 px. L'information est juste, mais trop bavarde : « Étape 4/5 · Transcription de l'audio », une phrase de détail, puis « Modèle déjà présent sur ce Mac · aucun téléchargement du modèle ». | [Vu dans le code] | Progress indicators | Ma | S | Une seule ligne principale, par exemple « Transcription… 03:12 ». Le détail du modèle n'apparaît que lors d'un **téléchargement**, avec sa taille. Le reste va dans un journal. | `desktop.py:249-278`, `desktop_worker.py:29-118` |

### Ergonomie

| ID | Constat | Statut | Réf. HIG | Grav. | Effort | Recommandation | Fichier:lignes |
|---|---|---|---|---|---|---|---|
| E1 | **La fenêtre fait au minimum 820×650 et empile 11 bandes**, soit environ 56 % × 68 % d'un MacBook Air 13" à la résolution par défaut. La taille est la même quand la fenêtre est vide. | Taille et empilement : [Vu dans le code]. Proportion de l'écran : [Hypothèse, calcul] | Windows | **B** | M | Créer un état vide compact d'environ 480×320 : zone de dépôt, plus Profil et Langue dans la barre d'outils. La fenêtre ne grandit que si l'historique est ouvert. Minimum d'environ 420×300. | `desktop.py:119-120` |
| E2 | **Le dépôt ne lance rien.** Il faut déposer, puis vérifier le profil, puis cliquer sur Transcrire. Pour un usage à un seul fichier, c'est un clic de trop et une ambiguïté : l'utilisateur se demande s'il s'est passé quelque chose. | [Vu dans le code] | Drag and drop | **B** | S | Lancer la transcription au dépôt avec le profil et la langue affichés. Exception : si OpenAI est activé, afficher une feuille de confirmation qui rappelle l'envoi de données. | `desktop.py:91-100`, `314-316` |
| E3 | La file (un tableau de 2 colonnes, plus « Retirer la sélection » et « Remettre en attente ») est toujours affichée alors que l'usage est à un fichier. Le terme « Remettre en attente » est du jargon. | [Vu dans le code] | Clarity | Ma | M | Supprimer le tableau. Un dépôt multiple met simplement les fichiers en attente, signalés par « 1 sur 3 » dans la ligne de statut. Le « réessayer » passe dans un menu contextuel de l'historique. | `desktop.py:140-145`, `216-233`, `241-247` |
| E4 | L'aperçu texte intégré et le lecteur « Lire en grand » (avec A−, A+ et plein écran recréés à la main) font doublon avec ce que l'utilisateur fait ailleurs. Dans le lecteur, **Échap ferme la fenêtre**, ce qui n'est pas standard (on attend Cmd+W). | [Vu dans le code] | Quick Look ; Keyboard | Ma | S | Supprimer l'aperçu et le lecteur. Montrer le résultat sur une carte avec trois actions : **Coup d'œil** (Espace), **Ouvrir** (application par défaut pour `.md`) et **Afficher dans le Finder**. | `desktop.py:26-67`, `228-247`, `389-400` |
| E5 | **Il n'y a pas de barre des menus applicative.** Aucune `QMenuBar` n'est créée ; seul le menu de l'app fourni par Qt existe probablement. Il manque Fichier (Ouvrir… Cmd+O, Ouvrir l'élément récent, Fermer Cmd+W), Édition, Présentation (afficher l'historique), Fenêtre et Aide. | Absence de `QMenuBar` : [Vu dans le code]. Menu de l'app fourni par Qt : [Hypothèse] | The menu bar | **B** | S (Qt) / S (SwiftUI) | Construire une barre complète : Fichier, Édition, Présentation, Fenêtre, Aide. En SwiftUI, une grande partie est fournie automatiquement (`CommandGroup`). | `desktop.py` (absent) |
| E6 | **Cmd+, ne fait rien.** Les réglages s'ouvrent par un bouton plat « Réglages… » dans le contenu, sous forme de feuille avec un bouton « Terminé ». Les champs sont désactivés pendant un traitement, sans explication. | [Vu dans le code] | Settings | Ma | S | Ouvrir une vraie fenêtre Réglages, non modale, par Cmd+, (scène `Settings` en SwiftUI, ou `QAction` avec `PreferencesRole` en Qt). Les changements s'appliquent immédiatement. Deux onglets : Général (format, destination, modèle) et Enrichissement (OpenAI). | `desktop.py:171-214` |
| E7 | **La langue et le format ont le même poids visuel que le profil**, sur une ligne de trois listes. Or le format change rarement, alors que la langue change de temps en temps (anglais). | [Vu dans le code] | Hierarchy | Ma | S | Mettre Profil et Langue dans des menus déroulants compacts de la barre d'outils, et le Format dans les Réglages. Ajouter Présentation > Langue > Français ⌘1, Anglais ⌘2, Auto ⌘0. | `desktop.py:148-169` |
| E8 | **Il n'y a pas d'historique.** La file est perdue à la fermeture, et un résultat d'hier n'est retrouvable que dans le Finder. | [Vu dans le code] | Sidebars | Ma | M | Ajouter une barre latérale repliable (Cmd+Ctrl+S), masquée par défaut. Chaque entrée affiche nom, date, durée, profil et état. Actions : Coup d'œil, Ouvrir, Finder, Retranscrire, Supprimer de l'historique. Signaler les fichiers déplacés au moyen de signets. | — |
| E9 | Les formats non pris en charge sont acceptés au survol, puis refusés **après** le dépôt par une alerte modale qui liste toutes les extensions. Un dossier sans fichier média produit la même alerte. | [Vu dans le code] | Drag and drop ; Alerts | Mi | S | Refuser le dépôt dès le survol (curseur d'interdiction). Si un dossier est mixte, afficher en ligne « 2 fichiers ignorés (format non pris en charge) », sans alerte. | `desktop.py:310-331`, `cli.py:124-144` |
| E10 | **Les messages d'erreur sont bruts.** Le texte de l'exception ou les 12 000 derniers caractères de stderr sont déversés dans l'aperçu, en tutoiement et sans accents (« Dependance manquante… Installe FFmpeg »). La consigne `brew install` n'a pas de sens pour un collègue. | [Vu dans le code] | Alerts ; Writing | Ma | M | Associer chaque erreur connue à un message en vouvoiement, avec une action : « FFmpeg introuvable » devient inutile si l'audio est décodé par AVFoundation (T5). Ajouter « Réseau indisponible : le modèle n'a pas pu être téléchargé. [Réessayer] ». Le reste va dans Aide > Afficher le journal. | `desktop.py:500-517`, `cli.py:95-122` |
| E11 | La fenêtre ne mémorise pas sa taille ni sa position (aucun `saveGeometry`/`restoreGeometry`). | [Vu dans le code] | Windows | Mi | S | En Qt : `saveGeometry` et `restoreGeometry` dans `QSettings`. En SwiftUI : `.defaultSize` plus la restauration d'état, qui est automatique. | `desktop.py:118-120` |
| E12 | **Le clavier est peu exploité.** Il n'y a pas de bouton par défaut (Entrée ne lance rien), pas de touche Suppr dans la file, pas d'Espace pour Coup d'œil, pas de Cmd+. pour arrêter. Cmd+O existe, mais comme action invisible. | [Vu dans le code] | Keyboard | Ma | S | Cmd+O pour Ouvrir, Cmd+. pour Arrêter, Espace pour Coup d'œil, Cmd+Entrée pour Ouvrir le résultat, Cmd+Retour arrière pour retirer de l'historique. Toutes ces commandes doivent figurer dans les menus. | `desktop.py:282-285` |
| E13 | L'icône de barre des menus (`QSystemTrayIcon`) est **toujours affichée**, avec une icône générique, et n'a pas de menu. Elle ne sert qu'à envoyer les notifications. | [Vu dans le code] | The menu bar (« extras ») | Ma | S | La supprimer. Envoyer les notifications par `UNUserNotificationCenter` (natif en Swift). Ajouter un badge sur l'icône du Dock pendant le traitement. | `desktop.py:279-281`, `438` |
| E14 | La notification de fin est envoyée même quand l'app est au premier plan, et elle n'est pas cliquable : elle ne permet pas d'ouvrir le résultat. | Envoi systématique : [Vu dans le code]. Notification non cliquable : [Hypothèse] | Notifications | Mi | S (SwiftUI) | Ne notifier que si l'app est en arrière-plan. Un clic sur la notification ouvre le résultat. | `desktop.py:438-439` |
| E15 | Le profil « Compte rendu » sans OpenAI produit un résumé **par règles** (détection de mots-clés), alors que le libellé suggère une synthèse. La mise en garde n'apparaît que dans les Réglages. | [Vu dans le code] | Writing (« be accurate ») | Ma | S | Renommer en « Compte rendu (structuré) », et afficher « Compte rendu enrichi par OpenAI » quand l'option est active. Ou ajouter une aide contextuelle dans le menu des profils. | `desktop.py:152`, `207` |
| E16 | La clé OpenAI doit être ressaisie à chaque session (choix de sécurité volontaire). | [Vu dans le code] | Privacy | Mi | S (SwiftUI) | La stocker dans le **Trousseau**, comme tu l'as accepté. | `desktop.py:203-206` |
| E17 | Les réglages ne sont enregistrés qu'au lancement d'un traitement ou à la fermeture. Après un plantage, un réglage modifié est perdu. | [Vu dans le code] | Settings | Mi | S | Enregistrer chaque réglage dès qu'il change (`@AppStorage`, ou `currentIndexChanged` en Qt). | `desktop.py:417-420`, `543-545` |
| E18 | Fermer la fenêtre pendant un traitement annule silencieusement le travail en cours, sans confirmation. | [Vu dans le code] | Alerts | Ma | S | Afficher une feuille : « Une transcription est en cours. Arrêter et fermer ? [Continuer] [Arrêter] ». Mieux : fermer la fenêtre sans quitter l'app, le traitement se poursuit et une notification arrive à la fin. | `desktop.py:536-541` |

### Accessibilité

| ID | Constat | Statut | Réf. HIG | Grav. | Effort | Recommandation | Fichier:lignes |
|---|---|---|---|---|---|---|---|
| A1 | Seuls deux contrôles ont un `setAccessibleName` (A− et A+). La zone de dépôt, la barre de progression, le chronomètre et les listes n'en ont pas ; les listes dépendent du `setBuddy` de leur libellé. | [Vu dans le code] | Accessibility ; VoiceOver | Ma | S | Nommer chaque contrôle. Annoncer les changements d'étape à VoiceOver (`accessibilityAnnouncement` en SwiftUI). Donner une valeur à la barre de progression. | `desktop.py:43-46`, `130`, `275` |
| A2 | Les couleurs codées en dur ne suivent ni « Augmenter le contraste » ni le mode sombre. Le texte désactivé (`#767b70` sur `#f0f1ee`) a un contraste d'environ 4,2:1, sous le seuil de 4,5:1. | Couleurs : [Vu dans le code]. Contraste : [Hypothèse, calcul] | Color ; Accessibility | Ma | S | Réglé par D1 et D2 : les couleurs système s'adaptent automatiquement. | `desktop.py:550-569` |
| A3 | Le texte est figé à 13 et 17 px. Le zoom n'existe que dans le lecteur, avec des boutons maison. | [Vu dans le code] | Typography | Mi | S | Utiliser les styles de texte système. Le zoom n'a plus de sens si le lecteur est supprimé (E4). | `desktop.py:37`, `551` |
| A4 | Il n'y a pas d'animation, donc rien à réduire. C'est conforme. | [Vu dans le code] | Motion | — | — | Si des transitions sont ajoutées en SwiftUI, respecter `accessibilityReduceMotion`. | — |
| A5 | Le focus clavier est dessiné à la main (`border: 2px solid #536432`) au lieu de l'anneau de focus système. | [Vu dans le code] | Keyboard ; Focus | Mi | S | Réglé par D1. | `desktop.py:558` |

### Technique

| ID | Constat | Statut | Réf. HIG | Grav. | Effort | Recommandation | Fichier:lignes |
|---|---|---|---|---|---|---|---|
| T1 | **Le `.app` n'est pas autonome.** Le lanceur C contient le chemin absolu du Python du `.venv` et le charge avec `dlopen`. Le code est copié dans `Resources`, mais pas les dépendances. Déplacer le dépôt ou le `.venv` casse l'app. | [Vu dans le code] | — | **B** (pour les collègues) | L | Embarquer un Python relocalisable (*python-build-standalone*) avec ses dépendances dans `Contents/Resources`. Le lancer comme processus de travail depuis une app SwiftUI (voir le lot 3). | `scripts/build_mac_app.py:28-41`, `scripts/mac_launcher.c` |
| T2 | **Pas de signature ni de notarisation, et tu n'as pas de compte développeur.** Sur Apple Silicon, une signature *ad hoc* (`codesign -s -`) suffit pour lancer l'app sur ton Mac. Chez un collègue, un `.app` téléchargé ou reçu par messagerie est mis en quarantaine, et Gatekeeper le bloque. Depuis macOS 15, le contournement par clic droit > Ouvrir ne marche plus : il faut aller dans Réglages Système > Confidentialité et sécurité > « Ouvrir quand même », mot de passe administrateur à l'appui. Les Mac de tes collègues sont fournis par l'entreprise, mais tu indiques que ce contournement y est autorisé. Autre effet de la signature ad hoc : l'identité de l'app change à chaque reconstruction, ce qui fait redemander l'accès au Trousseau (E16) et les autorisations d'accès aux dossiers. | Absence de signature : [Vu dans le code]. Comportement de Gatekeeper, du MDM et du Trousseau : [Hypothèse] | — | Ma (accepté) | S | **Choix assumé pour l'instant.** (1) Signer tout le bundle en ad hoc pendant le build (`codesign --force --deep -s -`), y compris les dylibs embarquées. (2) Fournir une fiche d'installation d'une page (Ouvrir quand même), avec en secours la commande `xattr -dr com.apple.quarantine`. (3) Le contournement est autorisé sur les Mac de tes collègues, mais un test réel reste utile : un premier `.app` signé ad hoc, envoyé à un collègue, pour s'assurer que la règle de l'entreprise est bien appliquée comme on la décrit. (4) Structurer le build pour que le passage en Developer ID et à la notarisation ne demande qu'une étape de plus, le jour où tu prendras un compte. | `scripts/build_mac_app.py` |
| T3 | Le PATH est forcé à `/opt/homebrew/bin` pour trouver FFmpeg, qui est un **prérequis externe** non embarqué. | [Vu dans le code] | — | **B** (pour les collègues) | M | Voir T5 : supprimer la dépendance à FFmpeg. | `desktop.py:460`, `scripts/mac_launcher.c:23-26`, `cli.py:114-121` |
| T4 | Le style natif de Qt ne donne ni barre d'outils unifiée au standard Apple, ni scène `Settings`, ni Coup d'œil (`QLPreviewPanel`), ni `UNUserNotificationCenter`, ni Trousseau, ni Services, ni Partager. Chacun demanderait du pont Objective-C. **C'est le plafond de la stack actuelle.** | Ce plafond est une limite connue de Qt Widgets : [Hypothèse] | — | Ma | — | Voir la décision d'architecture (lot 3). | — |
| T5 | Le décodage audio passe par le binaire `ffmpeg`, appelé par `mlx_whisper` à partir du chemin du fichier. | [Vu dans le code] (`run_whisper` passe un chemin) | — | Ma | M | Côté Swift : décoder avec **AVFoundation** (`AVAssetReader`) en PCM 16 kHz mono vers un WAV temporaire. Côté worker : lire ce WAV avec `wave` et numpy, puis passer le tableau à `mlx_whisper.transcribe`, qui accepte un `np.ndarray` [Hypothèse à vérifier]. FFmpeg n'est alors plus nécessaire. Limite : AVFoundation ne lit ni `.mkv`, ni `.webm`, ni `.ogg`, ni `.opus`, ni `.wma` [Hypothèse]. Il faut soit les refuser, soit embarquer un FFmpeg signé pour ces formats. | `cli.py:1451-1484`, `desktop_worker.py:113` |
| T6 | Un processus est lancé **par fichier** et le modèle est rechargé à chaque fois. Avec un seul fichier, c'est acceptable et même utile : la mémoire est libérée après le traitement. Mais lors d'un retraitement ou de dépôts successifs, chaque lancement coûte un rechargement. | Un processus par fichier : [Vu dans le code]. Durée du rechargement : [Hypothèse] | — | Mi | M | Garder ce comportement, mais faire vivre le worker quelques minutes après la fin d'un traitement avant de le fermer. | `desktop.py:457-475` |
| T7 | Le cache calcule le **SHA-256 du fichier entier** à l'étape 1. Pour une vidéo de 2 Go, cela représente plusieurs secondes de lecture, à chaque lancement. Le cache n'est jamais purgé. | [Vu dans le code] | — | Mi | S | Utiliser une clé rapide (taille, date de modification, inode, hachage des premiers et derniers Mo) et ne calculer le hachage complet qu'en cas de doute. Purger les entrées au-delà de N jours ou X Mo, et ajouter « Vider le cache » dans les Réglages. | `desktop_worker.py:57-66` |
| T8 | **Les modèles sont stockés dans `~/.cache/huggingface`**, un dossier caché, sans affichage de leur taille ni moyen de les supprimer. Le premier lancement télécharge environ 3 Go (large-v3) sans annoncer la taille ni montrer de progression. | Stockage et absence d'affichage : [Vu dans le code]. Taille : [Hypothèse] | Progress indicators | Ma | M | Stocker dans `~/Library/Application Support/Mp4ToTranscript/Models` (via `HF_HOME`). Ajouter dans les Réglages une section Modèles avec taille, emplacement et bouton Supprimer. Pour le téléchargement : afficher une progression **déterminée** en octets et demander confirmation avant les ~3 Go. Voir aussi la réserve sur large-v3 face à turbo, plus haut. | `desktop_worker.py:27-50`, `desktop.py:186` |
| T9 | **L'annulation est brutale** : SIGTERM, puis SIGKILL après 2,5 s. Le worker n'intercepte pas SIGTERM, donc les blocs `finally` ne s'exécutent pas. Un `.tmp` peut rester dans le cache, et un téléchargement Hugging Face interrompu laisse des fichiers `.incomplete`. | Code : [Vu dans le code]. Fichiers résiduels : [Hypothèse] | — | Mi | S | Intercepter SIGTERM dans le worker (lever `KeyboardInterrupt`) et nettoyer les `*.tmp` au démarrage. | `desktop.py:524-534`, `desktop_worker.py:69-78` |
| T10 | **Aucune progression réelle de la transcription.** La barre est indéterminée pendant plusieurs minutes. C'est honnête, mais ce n'est pas informatif. | [Vu dans le code] | Progress indicators | Ma | M | `mlx_whisper` traite le fichier par segments. En exploitant la sortie de chaque segment (`verbose` ou un callback), on obtient un pourcentage *réel* (temps du segment divisé par la durée obtenue avec ffprobe ou AVFoundation). Afficher aussi la progression sur l'icône du Dock. | `desktop_worker.py:111-115` |
| T11 | Le traitement ne bloque pas l'interface : il tourne dans un `QProcess` et l'interface lit les événements de façon asynchrone. **Point positif.** | [Vu dans le code] | — | — | — | Conserver ce protocole JSON-lines : c'est l'interface à garder entre SwiftUI et Python. | `desktop.py:457-503`, `desktop_worker.py` |
| T12 | Les exports n'écrasent jamais un fichier existant (ouverture en mode `x` avec suffixe numérique). Les tests couvrent le cache, l'annulation et les erreurs. **Point positif.** | [Vu dans le code] | — | — | — | Conserver. | `desktop_worker.py:81-91`, `tests/test_desktop.py` |
| T13 | **Pas de sandbox.** C'est acceptable hors App Store, mais l'historique demandera des signets pour rouvrir les fichiers dans le temps si un sandbox est ajouté plus tard. | Absence de sandbox : [Vu dans le code]. Besoin de signets : [Hypothèse] | — | Mi | — | Ne pas activer de sandbox pour la v1, distribuée hors App Store. Stocker quand même les chemins sous forme de signets (bookmarks), qui survivent aux déplacements de fichiers. | — |
| T14 | **Pas de mécanisme de mise à jour** : il faut reconstruire le lanceur à la main. | [Vu dans le code] | — | Mi | S | Pour 1 ou 2 collègues sans compte développeur, Sparkle n'en vaut pas la peine : chaque mise à jour repasserait de toute façon par Gatekeeper [Hypothèse]. Livrer un `.zip` versionné, sur GitHub Releases ou par partage direct, et afficher la version dans « À propos ». Prévoir Sparkle seulement avec la notarisation. | — |
| T15 | L'`Info.plist` est minimal. Il manque `LSMinimumSystemVersion`, `LSApplicationCategoryType`, `NSHumanReadableCopyright` et la restriction à arm64. La version est incohérente : `0.4.0` dans le plist, `0.1.0` dans `pyproject.toml`. | [Vu dans le code] | — | Mi | S | Compléter le plist et prendre une seule source pour la version. | `scripts/build_mac_app.py:43-53`, `pyproject.toml:7` |
| T16 | **Poids estimé.** Le `.app` actuel fait moins de 1 Mo, mais c'est trompeur. PySide6 pèse plusieurs centaines de Mo et mlx plus numpy environ 200 Mo. Les modèles pèsent environ 3 Go pour large-v3 et 1,6 Go pour turbo, avec plusieurs Go de mémoire unifiée pendant la transcription. | [Hypothèse] | — | Mi | — | Une version SwiftUI avec Python embarqué, **sans PySide6**, devrait peser environ 250 à 400 Mo, modèles non compris [Hypothèse]. Remplacer Qt par SwiftUI fait gagner du poids, pas en perdre. | — |

---

## 3. Plan d'action

### Lot 1 : gains rapides, en restant sur Qt (2 à 4 jours)

Ce lot apporte des gains même si la migration vers SwiftUI ne se fait pas, ou en attendant qu'elle se fasse.

1. **D1, D2, A2, A5 : rendre l'apparence native.** Supprimer Fusion, la palette et la feuille de style. C'est le plus gros gain visuel pour le plus petit effort. Il faut ensuite vérifier les modes clair et sombre sur des captures.
2. **E2 : lancer la transcription au dépôt**, avec confirmation si OpenAI est actif.
3. **E5, E6, E12 : ajouter les menus et les raccourcis.** Une `QMenuBar` avec Fichier, Édition, Présentation, Fenêtre et Aide. Les Réglages en `PreferencesRole` (Cmd+,). Cmd+. pour Arrêter.
4. **E13 : supprimer l'icône de barre des menus** et utiliser `QApplication.alert`, qui fait rebondir l'icône du Dock.
5. **D3, D7, E7 : alléger la fenêtre.** Retirer le titre et fusionner les trois lignes de statut en une. Envoyer le Format dans les Réglages.
6. **E11, E17, E18, T9 : corriger les petits défauts.** Mémoriser la géométrie de la fenêtre, enregistrer les réglages immédiatement, confirmer avant de fermer pendant un traitement, intercepter SIGTERM.
7. **E15 : renommer le profil** « Compte rendu (structuré) ».

### Lot 2 : améliorations structurantes, à faire de préférence directement en SwiftUI

1. **E1, E3, E4 : nouvelle mise en page compacte.**
   - Barre d'outils : Profil ▾, Langue ▾, ◧ Historique.
   - Contenu à **3 états** : dépôt → progression → carte de résultat avec Coup d'œil, Ouvrir et Finder.
   - Suppression du tableau, de l'aperçu et du lecteur.
2. **E8 : historique** dans une barre latérale repliable.
3. **T10 : progression réelle** en pourcentage, et progression sur l'icône du Dock.
4. **T8 : gestion des modèles.** Emplacement visible, taille, suppression, progression du téléchargement en octets.
5. **E10 : catalogue d'erreurs** en vouvoiement, chacune avec une action. Aide > Afficher le journal.
6. **E16, E14 : Trousseau et notifications natives**, cliquables.
7. **D6 : icône** refaite dans Icon Composer.
8. **T7 : clé de cache rapide** et purge du cache.

### Lot 3 : décision d'architecture

**Recommandation : SwiftUI pour l'interface, et le worker Python existant pour le moteur, en Python embarqué.**

**Pourquoi.** Tu veux une vraie app Mac transmissible à des collègues. Qt peut s'en approcher pour l'apparence, mais pas pour l'intégration (Réglages, Coup d'œil, Trousseau, notifications, icône macOS 26) ni pour le packaging. Le protocole JSON-lines entre interface et worker est déjà propre : il suffit de changer l'interface.

**Architecture cible**

```
Mp4ToTranscript.app
├── SwiftUI (macOS 14+, arm64)
│   ├── WindowGroup unique : NavigationSplitView (Historique | Dépôt → Progression → Résultat)
│   ├── Settings scene (Général · Modèles · Enrichissement)
│   ├── AVFoundation : décodage → WAV 16 kHz temporaire (remplace FFmpeg)
│   ├── Process + JSON-lines ↔ worker (protocole existant)
│   ├── SwiftData ou JSON : historique (+ bookmarks)
│   ├── Keychain (clé OpenAI), UNUserNotificationCenter, NSDockTile (progression)
│   └── (Sparkle plus tard, avec un compte développeur)
└── Resources/python/  (python-build-standalone + mlx-whisper + mp4_to_transcript)
    └── python -m mp4_to_transcript.desktop_worker
```

**Ce qui est conservé** : `cli.py`, avec tout le rendu et le compte rendu heuristique ; le protocole de `desktop_worker.py` ; les tests du worker. **Ce qui est supprimé** : `desktop.py` (640 lignes), `mac_launcher.c` et PySide6.

**Coût estimé** [Hypothèse], pour un développeur qui débute en Swift et s'appuie sur un assistant :

| Partie | Durée |
|---|---|
| Interface SwiftUI | 6 à 10 jours |
| Embarquement de Python et signature ad hoc (sans notarisation pour l'instant) | 3 à 5 jours |
| Historique, modèles, finitions | 4 à 7 jours |
| **Total** | **13 à 22 jours** |
| *Plus tard : passage en Developer ID et notarisation* | *2 à 4 jours, à prévoir le jour où tu prends un compte* |

**Risques**
1. **Distribution sans notarisation** : le premier lancement chez un collègue passe par « Ouvrir quand même ». Tu indiques que ce contournement est autorisé sur les Mac de tes collègues. Risque résiduel faible : une politique de l'entreprise peut changer. Vérifier une fois avec le premier build réel.
2. **Notariser plus tard un Python embarqué avec MLX** : chaque `.so` et `.dylib` devra être signé, et certains droits (entitlements) pourront être nécessaires. Ce risque est reporté, pas supprimé.
3. **Les formats que AVFoundation ne lit pas** (voir T5).
4. **Deux langages à maintenir.**

**Alternative à garder en tête, mais pas maintenant** : du 100 % Swift avec **WhisperKit** (CoreML), plus le portage en Swift des environ 1 500 lignes de rendu. C'est l'option la plus légère et la plus native, mais elle oblige à réécrire et à revalider tout le rendu. Elle ne se justifie que si l'embarquement de Python s'avère trop pénible à notariser.

**Ce que je déconseille** : investir dans la finition de la version Qt au-delà du lot 1. Ce serait du travail jeté.

---

## 4. Points positifs à préserver

Le fond est sain ; les reproches portent sur la forme et la distribution.

- **Séparation interface / worker** en JSON-lines, avec annulation et un processus isolé.
- **Exports qui n'écrasent jamais rien.**
- **Cache de transcription** : changer de rendu ne relance pas Whisper.
- **Honnêteté de l'interface** : pas de faux pourcentage, distinction affichée entre chargement et téléchargement du modèle.
- **OpenAI désactivé par défaut**, avec une mention explicite de l'envoi de données.
- **Tests du comportement de l'interface** (annulation, erreurs, doublons).
- **Ouverture depuis le Finder et le Dock**, testée par un vrai test d'intégration Launch Services.

---

## 5. Captures d'écran à fournir

À déposer dans `docs/audit/screenshots/`. Prendre des captures **système** (Cmd+Maj+4, puis Espace, puis cliquer sur la fenêtre) pour avoir la barre de titre, et ne pas utiliser `MP4_APP_SMOKE_SCREENSHOT`, qui ne capture que le contenu.

Chaque capture confirme ou infirme une hypothèse précise :

| # | Écran et état | Hypothèse à confirmer |
|---|---|---|
| 1 | Fenêtre vide, **mode clair** | E1, D3 (densité, titre) |
| 2 | Fenêtre vide, **mode sombre** | **D1** (contenu clair sous une barre de titre sombre ?) |
| 3 | Fenêtre réduite à sa **taille minimale**, sur ton écran réel | E1 (part de l'écran occupée) |
| 4 | Glisser un fichier **au-dessus** de la fenêtre, sans relâcher | D4 (aucun retour visuel ?) |
| 5 | Dépôt d'un fichier `.pdf`, ou d'un dossier sans média | E9 (alerte après coup) |
| 6 | Traitement en cours, **étape 4** | D7 (trois lignes de statut) |
| 7 | **Premier téléchargement** d'un modèle (cache Hugging Face vidé) | T8 (taille non annoncée, pas de progression) |
| 8 | Succès, avec le résultat sélectionné et l'aperçu | E4 |
| 9 | Erreur : renommer temporairement `ffmpeg`, ou couper le réseau pendant un téléchargement | E10 (message brut) |
| 10 | Après « Arrêter la file » | T9, E3 |
| 11 | Fenêtre **Réglages** | E6 (feuille ou fenêtre, bouton Terminé) |
| 12 | Lecteur « Lire en grand », en fenêtre puis en plein écran | E4 |
| 13 | **Barre des menus** de l'app, avec chaque menu ouvert | **E5** (quel menu Qt fournit réellement) |
| 14 | Icône dans la **barre des menus** (en haut à droite) et la **notification** de fin | E13, E14 |
| 15 | Icône dans le **Dock** à côté d'apps Apple, et dans le Finder (vue Icônes, grande taille) | **D6** (taille relative, fond gris sous macOS 26) |
| 16 | VoiceOver (Cmd+F5) sur la zone de dépôt et la barre de progression, avec le panneau de légende affiché | A1 |
| 17 | Réglages Système > Accessibilité > Affichage > « Augmenter le contraste » activé, fenêtre vide | A2 |

Indique aussi **la version de macOS** utilisée pour les captures : le rendu de l'icône et de la barre des menus en dépend.
