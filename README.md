# MotionBricks Project Webpage

Source for the MotionBricks (SIGGRAPH 2026) project page.

Target deployment: https://nvlabs.github.io/motionbricks
Dev repo (internal): https://gitlab-master.nvidia.com/tingwuw/motionbricks_webpage

## Preview locally

`python3 -m http.server` does **not** support HTTP range requests, so the large MP4s will stall. Use `http-server` (via `npx`) instead — it streams with `206 Partial Content`.

```bash
cd motionbricks_webpage
npx --yes http-server . -p 8000 -c-1 --cors
# open http://localhost:8000
```

## Clone

Requires **Git LFS**. Install once per machine: `git lfs install`.

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/tingwuw/motionbricks_webpage.git
```

## Layout

```
motionbricks_webpage/
├── index.html              main page — hero + 6 content sections + bibtex
├── static/
│   ├── css/
│   │   ├── bulma.min.css, bulma-carousel.min.css, bulma-slider.min.css
│   │   ├── fontawesome.all.min.css
│   │   └── index.css           page styles (NVIDIA Sans, green accent, Kimodo-style pill tabs)
│   ├── js/
│   │   ├── bulma-*.min.js, fontawesome.all.min.js
│   │   └── index.js            sticky-nav, hero playlist + toggle, lazy-load, pill-tab swap, single-play
│   ├── images/             teaser stills (LFS)
│   ├── posters/            first-frame thumbnails for every video (ffmpeg-generated)
│   ├── videos/             all MP4/MOV clips (LFS)
│   └── pdfs/
│       └── motionbricks_siggraph_2026.pdf  (LFS)
```

## Git LFS

`.gitattributes` tracks `*.mp4 *.mov *.webm *.pdf *.png`. Any new file with these extensions is auto-routed to LFS on commit.

Regenerate posters after adding new videos:

```bash
cd static
for v in videos/*.mp4 videos/*.mov videos/smart_locomotion/*.mp4 videos/smart_object/*.mp4; do
  out="posters/${v#videos/}"; out="${out%.*}.jpg"
  mkdir -p "$(dirname "$out")"
  ffmpeg -y -loglevel error -ss 0.2 -i "$v" -frames:v 1 -q:v 4 -vf "scale='min(1280,iw)':-2" "$out"
done
```

## Key features

- **Cinematic hero** — full-viewport background video (NVIDIA Sans wordmark overlaid). Ships two teasers; the "Animation / Robotics" pill-toggle directly under the button row lets visitors flip between them. Auto-advances to the other clip when the current one ends.
- **Lazy loading** — every non-hero `<video>` uses `preload="none"` + `data-lazy-video` + `<source data-src>`. An IntersectionObserver hydrates each one only when it scrolls near the viewport. Initial page weight is HTML + CSS + JS + poster JPGs.
- **Single-play** — when any page video starts, all others pause automatically.
- **Kimodo-style pill tabs** for Smart Locomotion (2 submodules: *Single Styles* and *Mixture of Styles*) and Smart Objects. Clicking a pill swaps both `<source>` and `poster`.
- **All media capped at `--media-max: 960px`** so every figure/video/image column-aligns.

## Pass 2 TODO

- [ ] Replace `#` placeholders for the arXiv and Code buttons with real URLs (Code currently points at `NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks`).
- [ ] Fill in author homepage links in `index.html`.
- [ ] Update BibTeX with the official ACM DOI once available.
- [ ] When `research.nvidia.com/labs/motionbricks/assets/` access lands, move the heaviest videos (`aibm_runtime_main_demo_latest_art_v3.mp4`, `motionbricks_demo_1080p30f.mp4`, `runtime_aibm_so_authoring_1.5x.mp4`) off the repo and onto the CDN — same pattern as GEAR-SONIC.
- [ ] Enable the sandbox-friendly LFS locks: `git config lfs.https://gitlab-master.nvidia.com/tingwuw/motionbricks_webpage.git/info/lfs.locksverify true` (optional, suppresses the noise on push).
