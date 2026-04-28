# ShotLab — AI Basketball Shooting Coach

## Project overview
AI-powered basketball shooting coach that analyzes jump shots from 3 smartphone camera angles (front, side, rear 45°) and gives personalized biomechanical feedback. Similar to Carv for skiing but camera-only, no hardware sensors.

## Current stack
- **Vanilla HTML/CSS/JS** — single `index.html` file for now, no build step
- **No backend** — videos stored in-memory via object URLs (local only)
- **No frameworks** — plain JS, no React/Vue yet
- **Fonts:** DM Mono, Barlow, Barlow Condensed (Google Fonts)

## Design system
```
Background:   #0c0d0f
Surface:      #13151a
Surface 2:    #1a1d24
Surface 3:    #20232c
Border:       #ffffff12
Accent:       #f97316  (orange)
Green:        #22c55e
Blue:         #3b82f6
Text:         #e8eaf0
Muted:        #6b7280
Subtle:       #9ca3af

Fonts:
  --mono:  'DM Mono', monospace
  --sans:  'Barlow', sans-serif
  --cond:  'Barlow Condensed', sans-serif  (used for headings + labels)
```

## Features built

### Feature 1 — 3-Angle Video Upload (`index.html`)
- 3 upload cards: Front View, Side View, Rear 45°
- Each card has an SVG court diagram showing camera placement
- Drag & drop or click-to-browse file upload
- Videos stored as object URLs (local, no server)
- Inline video preview per card with remove button
- Progress bar (0/3 → 1/3 → 2/3 → 3/3)
- "Analyze Session" button locked until all 3 loaded
- Setup tips row at the bottom
- Toast notifications

### Feature 2 — Synchronized 3-Video Scrubber (IN PROGRESS)
The player view is scaffolded. Needs completion:
- 3 video panels side by side (front / side / rear)
- Single scrubber that controls all 3 videos simultaneously
- Play/pause synced across all 3
- Playback speed: 0.25x, 0.5x, 1x
- Frame-by-frame step (← → arrow keys, also buttons)
- Per-panel zoom (click to expand one panel full width)
- Current time overlay on each panel
- Keyboard shortcuts: Space = play/pause, ← → = frame step, 1/2/3 = zoom panel

## Features to build next (in order)
1. Complete Feature 2 sync player JS logic
2. **Feature 3 — Shot detection:** auto-count shots in the video (ball tracking via color segmentation or YOLO API call)
3. **Feature 4 — Pose keypoint overlay:** draw skeleton over video using MediaPipe WASM or API
4. **Feature 5 — Analysis report:** arc score, drift, release consistency, one prioritized drill
5. **Feature 6 — Shot overlay:** stack all 10 shot trajectories on one canvas (the "wow" moment)

## Key product decisions
- **Primary user:** parents of youth players (10–17), not the player themselves
- **Primary B2B:** private trainers (they embed it with their clients)
- **Avoid:** overclaiming on hard-to-detect metrics (elbow flare, guide hand) — be conservative, build trust
- **Hero metric in V1:** arc angle (most reliable from video, most impactful coaching feedback)
- **Retention mechanic:** weekly shot streak + shareable "Shot Report" PNG card

## CV feasibility notes
| Metric | Reliability | Notes |
|---|---|---|
| Arc angle | ✅ Reliable | Ball tracking via color/YOLO |
| Release timing | ✅ Reliable | Wrist angle delta over frames |
| Left/right drift | ✅ Reliable | Ball trajectory vs body midline |
| Landing balance | ✅ Reliable | Pose estimation at landing |
| Shot pocket consistency | ✅ Reliable | Multi-frame pose |
| Base width | ⚠️ Env. dependent | Lighting affects leg keypoints |
| Torso lean | ⚠️ Angle dependent | Needs side view |
| Elbow flare | ❌ Hard | 2D ambiguity from single camera |
| Guide hand interference | ❌ Very hard | Needs high-res + perfect angle |

## File structure (current)
```
bball/
├── index.html      # Full app — upload view + player view scaffold
├── CLAUDE.md       # This file
└── README.md       # Project readme
```

## Running locally
No build step needed. Just open `index.html` in a browser.
For a local dev server: `npx serve .` or `python3 -m http.server 8080`

## When continuing in Claude Code
- Keep single-file architecture until Feature 4+ complexity demands splitting
- Maintain the dark aesthetic — no light mode, no generic UI
- Always test drag & drop AND click-to-browse for file upload
- The sync player must handle videos of different lengths gracefully (use shortest as master)
- Commit after each completed feature with message format: `feat: Feature N — description`
