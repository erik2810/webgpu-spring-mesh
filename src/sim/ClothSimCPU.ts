import * as THREE from 'three/webgpu';
import { attribute } from 'three/tsl';

import type { ClothSim } from '../core/types';
import type { SimParams } from '../core/params';
import { NEIGHBOURS, buildTopology, type Topology } from '../core/topology';

/** Constraint solves (Jacobi sweeps) per substep — must be even (ping-pong). */
const ITERATIONS = 2;

/**
 * CPU fallback solver for browsers without WebGPU (the renderer falls back to its
 * WebGL2 backend). Same small-step Position-Based Dynamics as {@link ClothSimGPU},
 * run in JS. Rendering uses a single triangulated wireframe grid whose per-vertex
 * `position`/`color` attributes are updated each frame — the most broadly
 * compatible path across the WebGPU and WebGL2 backends. `main.ts` caps the grid
 * density on this path so it still holds 60 fps.
 */
export class ClothSimCPU implements ClothSim {
  readonly object3D = new THREE.Group();
  readonly count: number;
  readonly cols: number;
  readonly rows: number;
  readonly springCount: number;
  readonly spacing: number;

  private readonly topo: Topology;
  private readonly pos: Float32Array; // current positions (bound to geometry)
  private readonly work: Float32Array; // ping-pong scratch for Jacobi sweeps
  private readonly prev: Float32Array;
  private readonly vel: Float32Array;
  private readonly vColor: Float32Array; // per-vertex colour (bound to geometry)
  private readonly fixed: Float32Array;

  private readonly geometry: THREE.BufferGeometry;

  private grab = -1;
  private readonly grabTarget = new THREE.Vector3();
  private t = 0;

  private gravity = 9.8;
  private stiffness = 0.9;
  private damping = 0.99;
  private mass = 1;
  private windStrength = 0;
  private windFreq = 2;
  private floorOn = false;
  private floorY = -2.6;
  private substeps = 8;
  private readonly speedScale = 4;

  constructor(params: SimParams) {
    const topo = buildTopology(params.density, params.density, params.spacing, params.pin);
    this.topo = topo;
    this.count = topo.count;
    this.cols = topo.cols;
    this.rows = topo.rows;
    this.springCount = topo.springCount;
    this.spacing = topo.spacing;
    this.fixed = topo.fixed;

    this.pos = topo.home.slice();
    this.work = topo.home.slice();
    this.prev = topo.home.slice();
    this.vel = new Float32Array(this.count * 3);
    this.vColor = new Float32Array(this.count * 3);

    // Triangulated grid index (two triangles per quad).
    const { cols, rows } = topo;
    const index = new Uint32Array((cols - 1) * (rows - 1) * 6);
    let t = 0;
    for (let y = 0; y < rows - 1; y++) {
      for (let x = 0; x < cols - 1; x++) {
        const i = y * cols + x;
        const r = i + 1;
        const d = i + cols;
        const dr = d + 1;
        index[t++] = i;
        index[t++] = r;
        index[t++] = d;
        index[t++] = r;
        index[t++] = dr;
        index[t++] = d;
      }
    }

    const geo = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(this.pos, 3);
    const colAttr = new THREE.BufferAttribute(this.vColor, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    colAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    geo.setAttribute('color', colAttr);
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.geometry = geo;

    const mat = new THREE.MeshBasicNodeMaterial({
      side: THREE.DoubleSide,
      wireframe: true,
      transparent: true,
      opacity: 0.92,
    });
    mat.colorNode = attribute('color') as any;

    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    this.object3D.add(mesh);

    this.setParams(params);
    this.refreshBuffers();
  }

  step(dt: number): void {
    const sdt = Math.min(dt, 1 / 30) / this.substeps;
    this.t += Math.min(dt, 1 / 30);
    for (let s = 0; s < this.substeps; s++) {
      this.predict(sdt);
      for (let it = 0; it < ITERATIONS; it++) this.solve();
      this.finalize(sdt);
    }
    this.refreshBuffers();
  }

  private locked(i: number): boolean {
    return this.fixed[i] > 0.5 || i === this.grab;
  }

  private clampFloor(y: number): number {
    return this.floorOn ? Math.max(y, this.floorY) : y;
  }

  private predict(sdt: number): void {
    const { pos, prev, vel, cols } = this;
    const invM = 1 / this.mass;
    for (let i = 0; i < this.count; i++) {
      prev[i * 3] = pos[i * 3];
      prev[i * 3 + 1] = pos[i * 3 + 1];
      prev[i * 3 + 2] = pos[i * 3 + 2];

      if (i === this.grab) {
        pos[i * 3] = this.grabTarget.x;
        pos[i * 3 + 1] = this.grabTarget.y;
        pos[i * 3 + 2] = this.grabTarget.z;
        prev[i * 3] = this.grabTarget.x;
        prev[i * 3 + 1] = this.grabTarget.y;
        prev[i * 3 + 2] = this.grabTarget.z;
        vel[i * 3] = vel[i * 3 + 1] = vel[i * 3 + 2] = 0;
        continue;
      }
      if (this.fixed[i] > 0.5) {
        vel[i * 3] = vel[i * 3 + 1] = vel[i * 3 + 2] = 0;
        continue;
      }
      const x = i % cols;
      const wind = this.windStrength * Math.sin(this.t * this.windFreq + x * 0.35) * invM;
      const vx = vel[i * 3];
      const vy = vel[i * 3 + 1] - this.gravity * sdt;
      const vz = vel[i * 3 + 2] + wind * sdt;
      vel[i * 3] = vx;
      vel[i * 3 + 1] = vy;
      vel[i * 3 + 2] = vz;
      pos[i * 3] += vx * sdt;
      pos[i * 3 + 1] = this.clampFloor(pos[i * 3 + 1] + vy * sdt);
      pos[i * 3 + 2] += vz * sdt;
    }
  }

  /** One Jacobi sweep: read `pos`, write `work`, then swap their contents. */
  private solve(): void {
    const { pos, work, cols, rows, spacing } = this;
    for (let i = 0; i < this.count; i++) {
      if (this.locked(i)) {
        work[i * 3] = pos[i * 3];
        work[i * 3 + 1] = pos[i * 3 + 1];
        work[i * 3 + 2] = pos[i * 3 + 2];
        continue;
      }
      const x = i % cols;
      const y = (i / cols) | 0;
      const px = pos[i * 3];
      const py = pos[i * 3 + 1];
      const pz = pos[i * 3 + 2];
      let cx = 0;
      let cy = 0;
      let cz = 0;
      let cnt = 0;
      for (const o of NEIGHBOURS) {
        const nx = x + o.dx;
        const ny = y + o.dy;
        if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
        const j = ny * cols + nx;
        const dx = pos[j * 3] - px;
        const dy = pos[j * 3 + 1] - py;
        const dz = pos[j * 3 + 2] - pz;
        const len = Math.hypot(dx, dy, dz) || 1e-4;
        const f = (len - o.rest * spacing) / len; // = (1 - rest/len)
        cx += dx * f;
        cy += dy * f;
        cz += dz * f;
        cnt++;
      }
      const k = cnt > 0 ? this.stiffness / cnt : 0;
      work[i * 3] = px + cx * k;
      work[i * 3 + 1] = this.clampFloor(py + cy * k);
      work[i * 3 + 2] = pz + cz * k;
    }
    for (let n = 0; n < this.pos.length; n++) {
      const tmp = this.pos[n];
      this.pos[n] = this.work[n];
      this.work[n] = tmp;
    }
  }

  private finalize(sdt: number): void {
    const { pos, prev, vel } = this;
    const inv = 1 / sdt;
    for (let i = 0; i < this.count; i++) {
      if (this.locked(i)) {
        vel[i * 3] = vel[i * 3 + 1] = vel[i * 3 + 2] = 0;
        continue;
      }
      vel[i * 3] = (pos[i * 3] - prev[i * 3]) * inv * this.damping;
      vel[i * 3 + 1] = (pos[i * 3 + 1] - prev[i * 3 + 1]) * inv * this.damping;
      vel[i * 3 + 2] = (pos[i * 3 + 2] - prev[i * 3 + 2]) * inv * this.damping;
    }
  }

  private refreshBuffers(): void {
    const { vel, vColor } = this;
    for (let i = 0; i < this.count; i++) {
      const sp = Math.min(
        1,
        Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]) / this.speedScale,
      );
      vColor[i * 3] = 0.13 + sp * 0.83; // teal → pink
      vColor[i * 3 + 1] = 0.83 - sp * 0.38;
      vColor[i * 3 + 2] = 0.93 - sp * 0.22;
    }
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
  }

  setParams(p: SimParams): void {
    this.gravity = p.gravity;
    this.stiffness = p.stiffness;
    this.damping = p.damping;
    this.mass = p.mass;
    this.windStrength = p.windStrength;
    this.windFreq = p.windFreq;
    this.floorOn = p.floorOn;
    this.floorY = p.floorY;
    this.substeps = Math.max(2, p.substeps - (p.substeps % 2));
  }

  setGrab(index: number, target: THREE.Vector3): void {
    this.grab = index;
    this.grabTarget.copy(target);
  }

  clearGrab(): void {
    this.grab = -1;
  }

  snapshotPositions(): Promise<Float32Array> {
    return Promise.resolve(this.pos);
  }

  reset(): void {
    this.grab = -1;
    this.t = 0;
    this.pos.set(this.topo.home);
    this.prev.set(this.topo.home);
    this.vel.fill(0);
    this.refreshBuffers();
  }

  dispose(): void {
    this.object3D.removeFromParent();
    this.geometry.dispose();
    this.object3D.traverse((obj) => {
      const mat = (obj as THREE.Mesh).material as THREE.Material | undefined;
      if (mat) mat.dispose();
    });
  }
}
