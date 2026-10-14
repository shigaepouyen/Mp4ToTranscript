# Desktop interface

French desktop utility. One window, drop area, saved profiles, queue, plain-text
preview, single primary action. Controls use the native macOS style: no custom
palette or stylesheet, so dark mode, the accent colour and Increase Contrast
follow the system. The only styled element is the drop area, drawn from the
system palette and highlighted while files are dragged over it.

Dropping or opening files starts the transcription with the displayed profile.
Sending text to OpenAI always asks for confirmation first. Controls that cannot
be used are hidden rather than shown disabled: progress, elapsed time and Stop
appear only while a file is processed; result actions only when a result is
selected. The OpenAI key field is disabled unless enrichment is on.

States use text: waiting, processing stage, completed, error, cancelled. The
progress indicator is indeterminate while Whisper runs; no invented percentage.
Advanced settings use a compact dedicated dialog to preserve queue and preview
space on small Mac screens. No decorative animation.

Long transcripts open in a separate resizable reader with zoom and full-screen
controls. Processing shows five numbered stages, elapsed time, and a persistent
model-cache status. Never imply a download when local model files are reused.

The planned native redesign (SwiftUI) is described in `docs/audit/audit-ux-ui.md`.
