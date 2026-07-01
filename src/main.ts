import './style.css';
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import WebGPU from 'three/addons/capabilities/WebGPU.js';

import { DEFAULTS, type SimParams } from './core/params';
import type { ClothSim } from './core/types';
import { ClothSimGPU } from './sim/ClothSimGPU';
import { ClothSimCPU } from './sim/ClothSimCPU';
import { PointerDragger } from './interaction/PointerDragger';
import { ControlPanel } from './ui/ControlPanel';

const loading = document.getElementById('loading');

function fail(message: string): void {
  if (!loading) return;
  loading.classList.remove('loading--hidden');
  loading.innerHTML = `<p class="loading__text" style="max-width:30ch;text-align:center;line-height:1.5">${message}</p>`;
}

async function bootstrap(): Promise<void> {
  // Backend (WebGPU vs WebGL2) and solver (GPU compute vs CPU) are independent.
  // The compute solver needs the WebGPU backend; the CPU solver runs on either.
  // `?cpu` forces the CPU solver even where WebGPU is available (for testing).
  const forceCPU = new URLSearchParams(location.search).has('cpu');
  const realWebGPU = WebGPU.isAvailable();
  const useGPUSim = realWebGPU && !forceCPU;
  // The CPU solver is JS-bound, so cap the grid (and substeps) to hold 60 fps.
  const maxDensity = useGPUSim ? 100 : 44;

  const params: SimParams = { ...DEFAULTS };
  if (!useGPUSim) {
    params.density = Math.min(params.density, maxDensity);
    params.substeps = Math.min(params.substeps, 4);
  }

  const renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL: !realWebGPU });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x05070d, 1);
  document.body.appendChild(renderer.domElement);

  try {
    await renderer.init();
  } catch (err) {
    console.error(err);
    fail('This demo needs a WebGPU- or WebGL2-capable browser. Try the latest Chrome, Edge, or Safari.');
    return;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0.6, 7.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2;
  controls.maxDistance = 24;

  // Faint reference grid, shown only with floor collision enabled.
  const grid = new THREE.GridHelper(24, 48, 0x24344f, 0x141d30);
  const gridMat = grid.material as THREE.Material;
  gridMat.transparent = true;
  gridMat.opacity = 0.5;
  grid.position.y = params.floorY;
  grid.visible = params.floorOn;
  scene.add(grid);

  const createSim = (): ClothSim =>
    useGPUSim ? new ClothSimGPU(renderer, params) : new ClothSimCPU(params);

  let sim = createSim();
  scene.add(sim.object3D);
  let dragger = new PointerDragger(renderer.domElement, camera, sim, controls);

  let paused = false;
  let lastFps = 0;
  let rebuildTimer = 0;

  const rebuild = (): void => {
    dragger.dispose();
    sim.dispose();
    sim = createSim();
    scene.add(sim.object3D);
    dragger = new PointerDragger(renderer.domElement, camera, sim, controls);
    panel.setStats({ fps: lastFps, nodes: sim.count, springs: sim.springCount });
  };

  const onParam = (p: SimParams, structural: boolean): void => {
    grid.position.y = p.floorY;
    grid.visible = p.floorOn;
    if (structural) {
      window.clearTimeout(rebuildTimer);
      rebuildTimer = window.setTimeout(rebuild, 140); // debounce while dragging density
    } else {
      sim.setParams(p);
    }
  };

  const panel = new ControlPanel(
    document.body,
    params,
    `${realWebGPU ? 'WebGPU' : 'WebGL'} · ${useGPUSim ? 'GPU compute' : 'CPU solver'}`,
    maxDensity,
    {
      onParam,
      onReset: () => sim.reset(),
      onPauseToggle: (p) => {
        paused = p;
      },
    },
  );
  panel.setBackend(useGPUSim ? 'WebGPU' : 'CPU');
  panel.setStats({ fps: 0, nodes: sim.count, springs: sim.springCount });

  // --- render loop + fps meter ---
  const timer = new THREE.Timer();
  let frames = 0;
  let accum = 0;

  renderer.setAnimationLoop(() => {
    timer.update();
    const dt = timer.getDelta();
    if (!paused) sim.step(dt);
    controls.update();
    renderer.render(scene, camera);

    frames++;
    accum += dt;
    if (accum >= 0.5) {
      lastFps = frames / accum;
      panel.setStats({ fps: lastFps, nodes: sim.count, springs: sim.springCount });
      frames = 0;
      accum = 0;
    }
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  if (loading) {
    loading.classList.add('loading--hidden');
    window.setTimeout(() => loading.remove(), 600);
  }

  // Dev-only manual driver (stripped from production builds): lets a headless
  // environment — where requestAnimationFrame is throttled — advance and render
  // the simulation deterministically for verification.
  if (import.meta.env.DEV) {
    (window as unknown as Record<string, unknown>).__mesh = {
      step: (n = 120) => {
        for (let i = 0; i < n; i++) sim.step(1 / 60);
      },
      render: () => renderer.render(scene, camera),
      reset: () => sim.reset(),
      grab: (i: number, x: number, y: number, z: number) =>
        sim.setGrab(i, new THREE.Vector3(x, y, z)),
      release: () => sim.clearGrab(),
      info: () => ({ nodes: sim.count, springs: sim.springCount }),
    };
  }
}

void bootstrap();
