/* ── Site-wide 3D wave background (VFX edition) ───────────────────────
   Whitish theme, greyish dots. GPU shader ocean: soft round particles,
   crest-shaded depth, cursor ripples with inertia, click shockwaves,
   ambient floating dust layer, scroll parallax.
   Fixed full-viewport canvas behind all content (Three.js via CDN,
   graceful fallback: page simply keeps its plain white background). */
(async () => {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  let THREE;
  try {
    THREE = await import('/three.module.min.js');
  } catch (e) { return; } // fallback: plain white page, no error

  const canvas = document.createElement('canvas');
  canvas.id = 'wave-bg';
  canvas.setAttribute('aria-hidden', 'true');
  canvas.style.cssText =
    'position:fixed;inset:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;display:block';
  document.body.prepend(canvas);
  // body backgrounds are solid #fff — make them transparent so the
  // canvas (sitting at z-index:-1) shows through, white stays on <html>
  document.documentElement.style.background = '#fff';
  document.body.style.background = 'transparent';

  const PR = Math.min(window.devicePixelRatio, 1.75);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(PR);
  renderer.setClearColor(0xffffff, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(58, 1, 0.05, 200);
  camera.position.set(0, 12, 28);
  camera.lookAt(0, 0, 0);

  // ── GPU shader ocean: displacement, ripples and shading on the GPU ──
  const COLS = 180, ROWS = 130, GAP = 0.46;
  const COUNT = COLS * ROWS;
  const positions = new Float32Array(COUNT * 3);
  const shades = new Float32Array(COUNT); // per-dot grain so the field isn't uniform
  let i3 = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      positions[i3]     = (c - COLS / 2) * GAP;
      positions[i3 + 1] = 0;
      positions[i3 + 2] = (r - ROWS / 2) * GAP;
      shades[i3 / 3] = Math.sin(c * 12.9898 + r * 78.233) * 0.5 + 0.5; // deterministic hash
      i3 += 3;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aShade', new THREE.BufferAttribute(shades, 1));

  const MAX_RIPPLES = 6;
  const uniforms = {
    uTime:        { value: 0 },
    uPR:          { value: PR },
    uMouse:       { value: new THREE.Vector2(0, 0) },
    uMouseActive: { value: 0 },
    uRipples:     { value: Array.from({ length: MAX_RIPPLES }, () => new THREE.Vector4(0, 0, -99, 0)) },
  };

  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false,
    vertexShader: /* glsl */`
      uniform float uTime;
      uniform float uPR;
      uniform vec2  uMouse;
      uniform float uMouseActive;
      uniform vec4  uRipples[${MAX_RIPPLES}]; // x, z, startTime, strength
      attribute float aShade;
      varying float vHeight;
      varying float vDist;
      varying float vShade;
      void main() {
        vec3 p = position;
        float d = length(p.xz);
        float y = sin(d * 0.55 - uTime * 1.8) * 1.15      // expanding rings
                + sin(p.x * 0.42 + uTime * 1.1) * 0.45    // cross swell
                + sin(p.z * 0.38 - uTime * 0.9) * 0.4;
        // cursor ripple (inertial uMouse smoothed on the CPU)
        float md = distance(p.xz, uMouse);
        y += uMouseActive * cos(min(md * 0.9, 3.14159)) * 2.4 * exp(-md * 0.22);
        // click shockwaves: expanding ring with travelling envelope
        for (int i = 0; i < ${MAX_RIPPLES}; i++) {
          vec4 rp = uRipples[i];
          float age = uTime - rp.z;
          if (rp.w > 0.0 && age > 0.0 && age < 4.0) {
            float rd = distance(p.xz, rp.xy);
            float env = exp(-abs(rd - age * 8.0) * 0.30) * exp(-age * 1.1);
            y += sin(rd * 1.7 - age * 10.0) * env * 3.2 * rp.w;
          }
        }
        p.y = y;
        vHeight = y;
        vShade = aShade;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vDist = -mv.z;
        gl_PointSize = uPR * (92.0 / vDist) * (1.05 + max(vHeight, 0.0) * 0.22);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      varying float vHeight;
      varying float vDist;
      varying float vShade;
      void main() {
        // soft round particle
        float r = length(gl_PointCoord - 0.5);
        float alpha = smoothstep(0.5, 0.16, r);
        // crests slightly darker, troughs lighter — soft light grey dots
        float shade = clamp(0.88 - vHeight * 0.03 - vShade * 0.04, 0.78, 0.95);
        // white "fog": far dots melt into the page background
        alpha *= (1.0 - smoothstep(16.0, 52.0, vDist)) * 0.92;
        gl_FragColor = vec4(vec3(shade), alpha);
      }`,
  });
  const ocean = new THREE.Points(geo, mat);
  scene.add(ocean);

  // ── ambient dust: sparse grey motes drifting in depth ──
  const spriteCanvas = document.createElement('canvas');
  spriteCanvas.width = spriteCanvas.height = 64;
  const sctx = spriteCanvas.getContext('2d');
  const grad = sctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(195,195,195,1)');
  grad.addColorStop(0.5, 'rgba(195,195,195,.35)');
  grad.addColorStop(1, 'rgba(195,195,195,0)');
  sctx.fillStyle = grad;
  sctx.fillRect(0, 0, 64, 64);
  const DUST = 260;
  const dustPos = new Float32Array(DUST * 3);
  for (let i = 0; i < DUST; i++) {
    const a = (i / DUST) * Math.PI * 2, rr = 8 + (i * 7919 % 100) / 100 * 30;
    dustPos[i * 3]     = Math.cos(a * 7.3) * rr;
    dustPos[i * 3 + 1] = 2 + (i * 104729 % 100) / 100 * 12;
    dustPos[i * 3 + 2] = Math.sin(a * 9.1) * rr;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({
    size: 0.55, map: new THREE.CanvasTexture(spriteCanvas), color: 0xc8c8c8,
    transparent: true, opacity: 0.32, depthWrite: false, sizeAttenuation: true,
  }));
  scene.add(dust);

  // ── input: inertial cursor, click shockwaves, scroll parallax ──
  const WORLD_X = COLS * GAP * 0.42, WORLD_Z = ROWS * GAP * 0.34;
  const mouse = { x: 0, y: 0, tx: 0, tz: 0, active: false };
  const toWorld = (cx, cy) => ({
    nx: (cx / window.innerWidth) * 2 - 1,
    ny: (cy / window.innerHeight) * 2 - 1,
  });
  window.addEventListener('mousemove', (e) => {
    const { nx, ny } = toWorld(e.clientX, e.clientY);
    mouse.x = nx; mouse.y = ny;
    mouse.tx = nx * WORLD_X; mouse.tz = ny * WORLD_Z;
    mouse.active = true;
  }, { passive: true });
  document.addEventListener('mouseleave', () => { mouse.active = false; });

  let rippleIdx = 0, timeNow = 0;
  window.addEventListener('pointerdown', (e) => {
    const { nx, ny } = toWorld(e.clientX, e.clientY);
    uniforms.uRipples.value[rippleIdx].set(nx * WORLD_X, ny * WORLD_Z, timeNow, 1);
    rippleIdx = (rippleIdx + 1) % MAX_RIPPLES;
  }, { passive: true });

  // scroll 0→1 across full page — camera dives inside waves by bottom, reverses on scroll up
  let scrollT = 0;
  function readScroll() {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    scrollT = max > 1 ? Math.min(window.scrollY / max, 1) : 0;
  }
  window.addEventListener('scroll', readScroll, { passive: true });
  readScroll();

  function resize() {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // ── animate ──
  let running = false, raf = 0;
  function animate(ts) {
    raf = requestAnimationFrame(animate);
    const t = ts / 1000;
    timeNow = t;
    uniforms.uTime.value = t;
    // inertial cursor: the ripple glides to the pointer instead of snapping
    const m = uniforms.uMouse.value;
    m.x += (mouse.tx - m.x) * 0.07;
    m.y += (mouse.tz - m.y) * 0.07;
    uniforms.uMouseActive.value += ((mouse.active ? 1 : 0) - uniforms.uMouseActive.value) * 0.05;

    ocean.rotation.y = Math.sin(t * 0.06) * 0.1;
    dust.rotation.y = t * 0.012;
    dust.position.y = Math.sin(t * 0.3) * 0.4;

    // camera: scroll drives zoom directly (high lerp = tracks scroll instantly)
    // mouse parallax is a soft additive layer on top
    // ease: slow start, fast plunge through waves at the end
    const dive = scrollT * scrollT;
    const scrollTargetY = 12 - dive * 17;   // y: 12 → -5 (punches below wave surface)
    const scrollTargetZ = 28 - dive * 25.5; // z: 28 → 2.5 (right inside)
    camera.position.x += ((mouse.active ? mouse.x * 2.6 : 0) - camera.position.x) * 0.05;
    camera.position.y = scrollTargetY - (mouse.active ? mouse.y * 1.0 : 0);
    camera.position.z = scrollTargetZ;
    // FOV blows wide open as you plunge inside
    camera.fov = 58 + dive * 42;
    camera.updateProjectionMatrix();
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  function start() { if (!running) { running = true; raf = requestAnimationFrame(animate); } }
  function stop()  { if (running)  { running = false; cancelAnimationFrame(raf); } }
  document.addEventListener('visibilitychange', () => {
    document.hidden ? stop() : start();
  });
  start();
})();
