import type * as THREE from 'three/webgpu';
import type { SimParams } from './params';

/**
 * Common surface implemented by both the WebGPU (compute-shader) backend and the
 * CPU (Verlet/JS) fallback. `main.ts` talks only to this interface, so the two
 * are fully interchangeable.
 */
export interface ClothSim {
  /** Renderable group, positioned at the world origin. */
  readonly object3D: THREE.Object3D;
  readonly count: number;
  readonly cols: number;
  readonly rows: number;
  readonly springCount: number;
  readonly spacing: number;

  /** Advance the simulation by `dt` seconds (internally substepped). */
  step(dt: number): void;

  /** Push live (non-structural) parameter values into the solver. */
  setParams(p: SimParams): void;

  /** Pin node `index` to `target` (world space) until {@link clearGrab}. */
  setGrab(index: number, target: THREE.Vector3): void;
  clearGrab(): void;

  /** Current node positions (xyz per node, world space) for picking. */
  snapshotPositions(): Promise<Float32Array>;

  /** Restore the rest configuration. */
  reset(): void;

  dispose(): void;
}
