# MotionBricks Project Webpage

Source for the MotionBricks (SIGGRAPH 2026) project page.

Deployed at: https://nvlabs.github.io/motionbricks

## Preview locally

`python3 -m http.server` does **not** support HTTP range requests, so large MP4s will stall. Use `http-server` (via `npx`) instead — it streams with `206 Partial Content`.

```bash
cd motionbricks_webpage
npx --yes http-server . -p 8000 -c-1 --cors
# open http://localhost:8000
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
│   ├── images/             teaser stills
│   ├── posters/            first-frame thumbnails for every video (ffmpeg-generated)
│   └── pdfs/
│       └── motionbricks_siggraph_2026.pdf
```

Videos are served from an external CDN and are not stored in this repository.

## Git LFS

`.gitattributes` tracks `*.pdf *.png`. Run `git lfs install` once per machine.

Regenerate posters after adding new videos (tries first frame, falls back to a later timestamp if the intro is mostly black):

```bash
cd static
yavg_at() {
  ffmpeg -ss "$2" -i "$1" -vframes 1 -vf "signalstats,metadata=print" -f null - 2>&1 \
    | grep -oE 'lavfi\.signalstats\.YAVG=[0-9.]+' | head -1 | cut -d= -f2
}
gen() {
  local src="$1" out="$2" t=0
  for cand in 0 5 10 15; do
    local y=$(yavg_at "$src" "$cand")
    if awk -v y="$y" 'BEGIN{exit !(y+0 >= 30)}'; then t=$cand; break; fi
    t=$cand
  done
  mkdir -p "$(dirname "$out")"
  ffmpeg -y -loglevel error -ss "$t" -i "$src" -frames:v 1 -q:v 4 -vf "scale='min(1280,iw)':-2" "$out"
}
# example: gen videos/foo.mp4 posters/foo.jpg
```

## Key features

- **Cinematic hero** — full-viewport background video (NVIDIA Sans wordmark overlaid). Ships two teasers; the "Animation / Robotics" pill-toggle directly under the button row lets visitors flip between them. Auto-advances to the other clip when the current one ends.
- **Lazy loading** — every non-hero `<video>` uses `preload="none"` + `data-lazy-video` + `<source data-src>`. An IntersectionObserver hydrates each one only when it scrolls near the viewport. Initial page weight is HTML + CSS + JS + poster JPGs.
- **Single-play** — when any page video starts, all others pause automatically.
- **Kimodo-style pill tabs** for Smart Locomotion (2 submodules: *Single Styles* and *Mixture of Styles*) and Smart Objects. Clicking a pill swaps both `<source>` and `poster`.
- **All media capped at `--media-max: 960px`** so every figure/video/image column-aligns.

## Pass 2 TODO

- [ ] Replace the `#` placeholder for the arXiv button with the real URL once assigned.
- [ ] Update BibTeX with the official ACM DOI once available.
