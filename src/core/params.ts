import type { PinMode } from './topology';

/**
 * All tunable parameters of the simulation. A single flat object is passed
 * around so the UI, the GPU sim, and the CPU sim all read the same source.
 */
export interface SimParams {
  /** Grid resolution; the mesh is `density x density` nodes. (structural) */
  density: number;
  /** Rest distance between adjacent nodes, in world units. (structural) */
  spacing: number;
  /** Which nodes are pinned in place. (structural) */
  pin: PinMode;

  /** Downward acceleration magnitude (m/s^2). */
  gravity: number;
  /** PBD constraint stiffness in [0,1] (relaxation factor): 1 = inextensible. */
  stiffness: number;
  /** Per-substep velocity retention (viscous drag). 1 = none, <1 = damped. */
  damping: number;
  /** Node mass (scales how wind translates into acceleration). */
  mass: number;

  /** Oscillating wind force amplitude (0 = still air). */
  windStrength: number;
  /** Wind temporal frequency. */
  windFreq: number;

  /** Floor-plane collision toggle. */
  floorOn: boolean;
  /** Height of the collision floor. */
  floorY: number;

  /** Integration substeps per frame (even values keep GPU ping-pong in buffer A). */
  substeps: number;
}

/** Keys that require rebuilding the mesh topology when changed. */
export const STRUCTURAL_KEYS = ['density', 'spacing', 'pin'] as const;

export const DEFAULTS: SimParams = {
  density: 56,
  spacing: 0.12,
  pin: 'top-corners',
  gravity: 9.8,
  stiffness: 0.9,
  damping: 0.99,
  mass: 1,
  windStrength: 0,
  windFreq: 2,
  floorOn: false,
  floorY: -2.6,
  substeps: 8,
};

export function isStructuralKey(key: keyof SimParams): boolean {
  return (STRUCTURAL_KEYS as readonly string[]).includes(key);
}
