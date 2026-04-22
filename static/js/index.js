document.addEventListener('DOMContentLoaded', () => {
  const nav = document.getElementById('site-nav');
  const toggle = nav?.querySelector('.site-nav-toggle');
  const navLinks = nav?.querySelector('.site-nav-links');

  if (toggle && navLinks) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    navLinks.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        nav.classList.remove('is-open');
        toggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  const scrollThreshold = Math.max(window.innerHeight * 0.6, 200);
  const updateNavTint = () => {
    if (!nav) return;
    if (window.scrollY > scrollThreshold) {
      nav.classList.add('is-scrolled');
    } else {
      nav.classList.remove('is-scrolled');
    }
  };
  window.addEventListener('scroll', updateNavTint, { passive: true });
  updateNavTint();

  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Hero playlist — shuffled on load, plays clips back-to-back. Optional toggle UI.
  const heroVideo = document.getElementById('heroVideo');
  if (heroVideo && heroVideo.dataset.playlist) {
    let playlist = [];
    try { playlist = JSON.parse(heroVideo.dataset.playlist); } catch (e) { playlist = []; }
    // normalise: allow both ["url", ...] and [{src,label,poster}, ...]
    playlist = (Array.isArray(playlist) ? playlist : []).map(item =>
      typeof item === 'string' ? { src: item, label: '', poster: '' } : item
    );

    if (playlist.length) {
      // Play in declared order — first entry is the default.
      const order = playlist.map((_, i) => i);
      let cursor = 0;

      const applyClip = (pIdx, userInitiated) => {
        const item = playlist[pIdx];
        if (!item) return;
        if (item.poster) heroVideo.setAttribute('poster', item.poster);
        heroVideo.src = item.src;
        heroVideo.load();
        const pr = heroVideo.play();
        if (pr && typeof pr.catch === 'function') pr.catch(() => {});
        // Reflect active toggle state.
        document.querySelectorAll('#heroBgToggle .hero-bg-btn').forEach(btn => {
          btn.classList.toggle('is-active', Number(btn.dataset.index) === pIdx);
        });
        if (userInitiated) {
          // Align the auto-advance cursor so the NEXT clip comes after this pick.
          const oIdx = order.indexOf(pIdx);
          if (oIdx >= 0) cursor = oIdx;
        }
      };

      const playNext = () => {
        const pIdx = order[cursor % order.length];
        cursor = (cursor + 1) % order.length;
        applyClip(pIdx, false);
      };
      heroVideo.addEventListener('ended', playNext);

      // Build toggle buttons only when there are at least 2 clips with labels.
      const toggleHost = document.getElementById('heroBgToggle');
      if (toggleHost && playlist.length > 1 && playlist.some(p => p.label)) {
        // preserve any pre-rendered .hero-bg-label child, drop any prior buttons
        toggleHost.querySelectorAll('.hero-bg-btn').forEach(b => b.remove());
        playlist.forEach((item, i) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'hero-bg-btn';
          btn.textContent = item.label || `Clip ${i + 1}`;
          btn.dataset.index = String(i);
          btn.addEventListener('click', () => applyClip(i, true));
          toggleHost.appendChild(btn);
        });
      }

      // Kick off playback with the first shuffled clip.
      applyClip(order[0], false);
      cursor = 1;
    }
  }

  const hydrateLazyVideo = (video) => {
    if (video.dataset.hydrated === 'true') return false;
    let changed = false;
    video.querySelectorAll('source[data-src]').forEach(s => {
      if (!s.src) {
        s.src = s.dataset.src;
        changed = true;
      }
    });
    if (changed) {
      video.load();
      video.dataset.hydrated = 'true';
    }
    return changed;
  };

  // Hydrate data-src once a lazy video scrolls near the viewport — but don't autoplay.
  const lazyVideos = document.querySelectorAll('video[data-lazy-video]');
  if (lazyVideos.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          hydrateLazyVideo(entry.target);
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: '300px 0px', threshold: 0.01 });
    lazyVideos.forEach(v => io.observe(v));
  }

  // Single-play: when any page video starts, pause every other page video.
  // Excludes the hero background video (which plays its shuffled playlist on loop).
  const pageVideos = Array.from(document.querySelectorAll('video'));
  pageVideos.forEach(v => {
    if (v === heroVideo) return;
    v.addEventListener('play', () => {
      pageVideos.forEach(other => {
        if (other !== v && other !== heroVideo && !other.paused) {
          other.pause();
        }
      });
    });
  });

  // Pill tabs: swap source + poster, hydrate, and play on click.
  document.querySelectorAll('.pill-tabs').forEach(tabs => {
    const videoId = tabs.dataset.videoTarget;
    const video = videoId ? document.getElementById(videoId) : null;
    if (!video) return;

    tabs.addEventListener('click', (e) => {
      const pill = e.target.closest('.pill-tab');
      if (!pill) return;

      const wasActive = pill.classList.contains('is-active');
      tabs.querySelectorAll('.pill-tab').forEach(p => p.classList.remove('is-active'));
      pill.classList.add('is-active');

      const src = pill.dataset.src;
      const poster = pill.dataset.poster;

      if (!wasActive && src) {
        video.style.opacity = '0';
        window.setTimeout(() => {
          if (poster) video.setAttribute('poster', poster);
          const source = video.querySelector('source');
          if (source) {
            source.src = src;
            source.removeAttribute('data-src');
          } else {
            video.src = src;
          }
          video.dataset.hydrated = 'true';
          video.load();
          const p = video.play();
          if (p && typeof p.catch === 'function') p.catch(() => {});
          video.style.opacity = '1';
        }, 180);
      } else {
        const p = video.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      }
    });
  });
});
