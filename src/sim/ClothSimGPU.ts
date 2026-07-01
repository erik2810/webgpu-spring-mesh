import * as THREE from 'three/webgpu';
import {
  Fn,
  If,
  attribute,
  cameraPosition,
  color,
  cross,
  float,
  instanceIndex,
  int,
  max,
  mix,
  normalize,
  normalWorld,
  positionLocal,
  sin,
  smoothstep,
  storage,
  time,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';

import type { ClothSim } from '../core/types';
import type { SimParams } from '../core/params';
import { NEIGHBOURS, buildTopology, type Topology } from '../core/topology';

// TSL node objects are only loosely modelled in the published d.ts (storage
// elements expose no swizzles, FnNode has no `.compute`), so the node layer is
// aliased to `any`. The shader graph is validated at runtime by the WGSL
// compiler — TypeScript adds little here.
type Storage = any;
type Compute = any;
type Attr = THREE.StorageInstancedBufferAttribute;

/** Constraint solves (Jacobi sweeps) per substep. */
const ITERATIONS = 2;

/**
 * GPU cloth solver using small-step Position-Based Dynamics (Müller 2007;
 * Macklin "Small Steps in Physics Simulation" 2019).
 *
 * Each substep runs three compute passes:
 *   1. predict  — integrate gravity/wind into velocity, advance positions.
 *   2. solve    — Jacobi distance-constraint projection (ping-ponged), repeated
 *                 `ITERATIONS` times, pulling every spring toward its rest length.
 *   3. finalize — recover velocity from the position delta, write speed for shading.
 *
 * PBD is unconditionally stable: "stiffness" is a 0–1 relaxation factor, so no
 * slider combination can blow the system up. Storage nodes are created fresh per
 * kernel so WebGPU access modes are inferred per-pipeline (sharing one node
 * across a read- and a write-kernel makes TSL mark it read-only and reject the
 * write).
 */
export class ClothSimGPU implements ClothSim {
  readonly object3D = new THREE.Group();
  readonly count: number;
  readonly cols: number;
  readonly rows: number;
  readonly springCount: number;
  readonly spacing: number;

  private readonly renderer: THREE.WebGPURenderer;
  private readonly topo: Topology;

  private readonly posA: Attr; // ping-pong position buffers (xyz, w = speed)
  private readonly posB: Attr;
  private readonly prevAttr: Attr; // position before prediction
  private readonly velAttr: Attr;
  private readonly fixedAttr: Attr;

  private readonly kPredict: Compute;
  private readonly kSolveAB: Compute;
  private readonly kSolveBA: Compute;
  private readonly kFinalize: Compute;

  private readonly U: Record<string, any>;
  private substeps = 8;

  private readonly nodeMesh: THREE.InstancedMesh;
  private readonly springMesh: THREE.InstancedMesh;
  private readonly snap: Float32Array;

  constructor(renderer: THREE.WebGPURenderer, params: SimParams) {
    this.renderer = renderer;
    const topo = buildTopology(params.density, params.density, params.spacing, params.pin);
    this.topo = topo;
    this.count = topo.count;
    this.cols = topo.cols;
    this.rows = topo.rows;
    this.springCount = topo.springCount;
    this.spacing = topo.spacing;
    this.snap = topo.home.slice();

    const { count, cols, rows, spacing } = topo;

    this.posA = new THREE.StorageInstancedBufferAttribute(count, 4);
    this.posB = new THREE.StorageInstancedBufferAttribute(count, 4);
    this.prevAttr = new THREE.StorageInstancedBufferAttribute(count, 4);
    this.velAttr = new THREE.StorageInstancedBufferAttribute(count, 4);
    this.fixedAttr = new THREE.StorageInstancedBufferAttribute(count, 1);
    (this.fixedAttr.array as Float32Array).set(topo.fixed);
    this.uploadRestState();

    this.U = {
      gravity: uniform(new THREE.Vector3(0, -params.gravity, 0)),
      stiffness: uniform(params.stiffness),
      damping: uniform(params.damping),
      mass: uniform(params.mass),
      windDir: uniform(new THREE.Vector3(0, 0, 1)),
      windStrength: uniform(params.windStrength),
      windFreq: uniform(params.windFreq),
      dt: uniform(1 / 480),
      grabIndex: uniform(-1),
      grabTarget: uniform(new THREE.Vector3()),
      floorOn: uniform(params.floorOn ? 1 : 0),
      floorY: uniform(params.floorY),
      nodeRadius: uniform(spacing * 0.17),
      lineWidth: uniform(spacing * 0.05),
      speedScale: uniform(4.0),
    };
    const U = this.U;

    const lockedAt = (idx: any, fixedNode: Storage): any =>
      fixedNode.element(instanceIndex).greaterThan(0.5).or(float(idx).equal(U.grabIndex));

    const clampFloor = (p: any): any =>
      vec3(p.x, U.floorOn.greaterThan(0.5).select(max(p.y, U.floorY), p.y), p.z);

    // --- predict: gravity + wind → velocity, advance positions ------------
    this.kPredict = (
      Fn(() => {
        const pos: Storage = storage(this.posA, 'vec4', count);
        const prev: Storage = storage(this.prevAttr, 'vec4', count);
        const vel: Storage = storage(this.velAttr, 'vec4', count);
        const fixedNode: Storage = storage(this.fixedAttr, 'float', count);

        const idx = int(instanceIndex);
        const xx = idx.sub(idx.div(int(cols)).mul(int(cols)));
        const po = pos.element(instanceIndex).xyz.toVar();
        const vo = vel.element(instanceIndex).xyz.toVar();

        const isGrab = float(idx).equal(U.grabIndex);
        const isFixed = fixedNode.element(instanceIndex).greaterThan(0.5);

        If(isGrab, () => {
          prev.element(instanceIndex).assign(vec4(U.grabTarget, 0));
          pos.element(instanceIndex).assign(vec4(U.grabTarget, 0));
          vel.element(instanceIndex).assign(vec4(0));
        })
          .ElseIf(isFixed, () => {
            prev.element(instanceIndex).assign(vec4(po, 0));
            vel.element(instanceIndex).assign(vec4(0));
          })
          .Else(() => {
            const phase = time.mul(U.windFreq).add(float(xx).mul(0.35));
            const wind = U.windDir.mul(U.windStrength.mul(sin(phase))).div(U.mass);
            const vn = vo.add(U.gravity.add(wind).mul(U.dt));
            const pn = clampFloor(po.add(vn.mul(U.dt)));
            prev.element(instanceIndex).assign(vec4(po, 0));
            vel.element(instanceIndex).assign(vec4(vn, 0));
            pos.element(instanceIndex).assign(vec4(pn, 0));
          });
      })() as any
    ).compute(count);

    // --- solve: one Jacobi sweep of distance constraints ------------------
    const makeSolve = (readAttr: Attr, writeAttr: Attr): Compute =>
      (
        Fn(() => {
          const read: Storage = storage(readAttr, 'vec4', count);
          const write: Storage = storage(writeAttr, 'vec4', count);
          const fixedNode: Storage = storage(this.fixedAttr, 'float', count);

          const idx = int(instanceIndex);
          const yy = idx.div(int(cols));
          const xx = idx.sub(yy.mul(int(cols)));
          const self = read.element(instanceIndex).toVar();

          If(lockedAt(idx, fixedNode), () => {
            write.element(instanceIndex).assign(self);
          }).Else(() => {
            const p = self.xyz.toVar();
            const corr = vec3(0, 0, 0).toVar();
            const cnt = float(0).toVar();

            for (const o of NEIGHBOURS) {
              const nx = xx.add(int(o.dx));
              const ny = yy.add(int(o.dy));
              const inBounds = nx
                .greaterThanEqual(int(0))
                .and(nx.lessThan(int(cols)))
                .and(ny.greaterThanEqual(int(0)))
                .and(ny.lessThan(int(rows)));
              If(inBounds, () => {
                const np = read.element(ny.mul(int(cols)).add(nx)).xyz;
                const d = np.sub(p);
                const len = d.length().max(1e-4);
                const rest = float(o.rest * spacing);
                // Move toward the neighbour proportional to the rest-length error.
                corr.addAssign(d.div(len).mul(len.sub(rest)));
                cnt.addAssign(1);
              });
            }

            const moved = p.add(corr.div(cnt.max(1)).mul(U.stiffness));
            write.element(instanceIndex).assign(vec4(clampFloor(moved), self.w));
          });
        })() as any
      ).compute(count);

    this.kSolveAB = makeSolve(this.posA, this.posB);
    this.kSolveBA = makeSolve(this.posB, this.posA);

    // --- finalize: velocity from the position delta, speed for shading ----
    this.kFinalize = (
      Fn(() => {
        const pos: Storage = storage(this.posA, 'vec4', count);
        const prev: Storage = storage(this.prevAttr, 'vec4', count);
        const vel: Storage = storage(this.velAttr, 'vec4', count);
        const fixedNode: Storage = storage(this.fixedAttr, 'float', count);

        const idx = int(instanceIndex);
        const p = pos.element(instanceIndex).xyz.toVar();

        If(lockedAt(idx, fixedNode), () => {
          vel.element(instanceIndex).assign(vec4(0));
          pos.element(instanceIndex).assign(vec4(p, 0));
        }).Else(() => {
          const v = p.sub(prev.element(instanceIndex).xyz).div(U.dt).mul(U.damping);
          vel.element(instanceIndex).assign(vec4(v, 0));
          pos.element(instanceIndex).assign(vec4(p, v.length()));
        });
      })() as any
    ).compute(count);

    // Render materials read buffer A through a dedicated read-only storage node.
    const posRenderBase: Storage = storage(this.posA, 'vec4', count);
    const posRender: Storage =
      typeof posRenderBase.toReadOnly === 'function' ? posRenderBase.toReadOnly() : posRenderBase;

    // --- node instances ---------------------------------------------------
    const nodeMat = new THREE.MeshBasicNodeMaterial();
    const nodePos = posRender.element(instanceIndex);
    nodeMat.positionNode = nodePos.xyz.add(positionLocal.mul(U.nodeRadius));
    {
      const speed = smoothstep(float(0), U.speedScale, nodePos.w);
      const base = mix(color(0x22d3ee), color(0xf472b6), speed);
      const view = normalize(cameraPosition.sub(nodePos.xyz));
      const fres = float(1).sub(normalWorld.dot(view).clamp(0, 1)).pow(2.5);
      nodeMat.colorNode = base.add(color(0xffffff).mul(fres.mul(0.5)));
    }
    this.nodeMesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), nodeMat, count);
    this.nodeMesh.frustumCulled = false;

    // --- spring instances (screen-facing ribbons) -------------------------
    const ends = new Float32Array(this.springCount * 2);
    ends.set(topo.springs);
    const quad = new THREE.BufferGeometry();
    quad.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, 1, 0], 3),
    );
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    quad.setAttribute('aEnds', new THREE.InstancedBufferAttribute(ends, 2));
    quad.setAttribute('aRest', new THREE.InstancedBufferAttribute(topo.springRest.slice(), 1));

    const springMat = new THREE.MeshBasicNodeMaterial({
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.92,
    });
    {
      const e: any = attribute('aEnds');
      const A = posRender.element(int(e.x)).xyz;
      const B = posRender.element(int(e.y)).xyz;
      const seg = B.sub(A);
      const len = seg.length().max(1e-4);
      const mid = mix(A, B, positionLocal.x);
      const view = normalize(cameraPosition.sub(mid));
      const side = normalize(cross(seg.div(len), view)).mul(U.lineWidth.mul(positionLocal.y));
      springMat.positionNode = mid.add(side);

      const strain = len.sub(attribute('aRest')).div(attribute('aRest'));
      const stretch = smoothstep(float(0), float(0.18), strain);
      const compress = smoothstep(float(0), float(0.18), strain.negate());
      springMat.colorNode = mix(
        mix(color(0x22d3ee), color(0xfb7185), stretch),
        color(0x3b82f6),
        compress,
      );
    }
    this.springMesh = new THREE.InstancedMesh(quad, springMat, this.springCount);
    this.springMesh.frustumCulled = false;

    this.object3D.add(this.springMesh, this.nodeMesh);
    this.setParams(params);
  }

  private uploadRestState(): void {
    const home = this.topo.home;
    for (const attr of [this.posA, this.posB, this.prevAttr]) {
      const a = attr.array as Float32Array;
      for (let i = 0; i < this.count; i++) {
        a[i * 4 + 0] = home[i * 3 + 0];
        a[i * 4 + 1] = home[i * 3 + 1];
        a[i * 4 + 2] = home[i * 3 + 2];
        a[i * 4 + 3] = 0;
      }
      attr.needsUpdate = true;
    }
    (this.velAttr.array as Float32Array).fill(0);
    this.velAttr.needsUpdate = true;
  }

  step(dt: number): void {
    this.U.dt.value = Math.min(dt, 1 / 30) / this.substeps;
    for (let s = 0; s < this.substeps; s++) {
      void this.renderer.computeAsync(this.kPredict);
      for (let it = 0; it < ITERATIONS; it++) {
        void this.renderer.computeAsync(it % 2 === 0 ? this.kSolveAB : this.kSolveBA);
      }
      void this.renderer.computeAsync(this.kFinalize);
    }
  }

  setParams(p: SimParams): void {
    this.U.gravity.value.set(0, -p.gravity, 0);
    this.U.stiffness.value = p.stiffness;
    this.U.damping.value = p.damping;
    this.U.mass.value = p.mass;
    this.U.windStrength.value = p.windStrength;
    this.U.windFreq.value = p.windFreq;
    this.U.floorOn.value = p.floorOn ? 1 : 0;
    this.U.floorY.value = p.floorY;
    this.substeps = Math.max(2, p.substeps - (p.substeps % 2));
  }

  setGrab(index: number, target: THREE.Vector3): void {
    this.U.grabIndex.value = index;
    this.U.grabTarget.value.copy(target);
  }

  clearGrab(): void {
    this.U.grabIndex.value = -1;
  }

  async snapshotPositions(): Promise<Float32Array> {
    try {
      const ab = await this.renderer.getArrayBufferAsync(this.posA);
      const v4 = new Float32Array(ab);
      for (let i = 0; i < this.count; i++) {
        this.snap[i * 3 + 0] = v4[i * 4 + 0];
        this.snap[i * 3 + 1] = v4[i * 4 + 1];
        this.snap[i * 3 + 2] = v4[i * 4 + 2];
      }
    } catch {
      /* readback unsupported — fall back to last-known positions */
    }
    return this.snap;
  }

  reset(): void {
    this.clearGrab();
    this.uploadRestState();
    this.snap.set(this.topo.home);
  }

  dispose(): void {
    this.object3D.removeFromParent();
    this.nodeMesh.geometry.dispose();
    (this.nodeMesh.material as THREE.Material).dispose();
    this.springMesh.geometry.dispose();
    (this.springMesh.material as THREE.Material).dispose();
  }
}
