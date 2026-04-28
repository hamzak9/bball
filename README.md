# ShotLab 🏀

AI-powered basketball shooting coach that analyzes jump shots from 3 smartphone camera angles and provides personalized biomechanical feedback.

## Status
Early build — feature by feature.

| Feature | Status |
|---|---|
| F1 — 3-angle video upload | ✅ Done |
| F2 — Synchronized 3-video scrub player | ✅ Done |
| F3 — Shot detection (ball tracking) | 🔜 Next |
| F4 — Pose keypoint overlay | 🔜 Planned |
| F5 — Analysis report | 🔜 Planned |
| F6 — Shot trajectory overlay | 🔜 Planned |

## Running locally
No build step. Open `index.html` in any modern browser.

```bash
# optional local server
npx serve .
# or
python3 -m http.server 8080
```

## Stack
- Vanilla HTML / CSS / JS — single file
- Videos stored as in-memory object URLs (no backend yet)
- Google Fonts: DM Mono, Barlow, Barlow Condensed

## Context for AI assistants
See `CLAUDE.md` for full product context, design system, and feature roadmap.
