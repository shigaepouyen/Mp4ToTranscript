# Desktop interface

French desktop utility, system typography and standard Qt controls. One window,
drop area, saved profiles, queue, plain-text preview, single primary action.

Palette source (OKLCH): primary `oklch(0.40 0.07 120)`, background
`oklch(0.985 0 0)`, surface `oklch(1 0 0)`, ink `oklch(0.25 0.01 120)`,
muted `oklch(0.40 0.025 120)`. Qt stylesheets use corresponding sRGB hex values
because Qt does not parse OKLCH. Accent appears on actions and progress only.

States use text: waiting, processing stage, completed, error, cancelled. The
progress indicator is indeterminate while Whisper runs; no invented percentage.
Advanced settings use a compact dedicated dialog to preserve queue and preview
space on small Mac screens. No decorative animation.
