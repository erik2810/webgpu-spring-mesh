/**
 * Mesh topology for a structured spring lattice.
 *
 * Nodes are laid out on a regular `cols x rows` grid, index = y * cols + x.
 * Springs follow the classic mass-spring cloth model (Provot 1995):
 *
 *   - structural springs connect 4-neighbours          (rest = h)
 *   - shear springs connect diagonal neighbours         (rest = h * sqrt 2)
 *   - bend springs connect 2-away neighbours            (rest = 2h)
 *
 * `NEIGHBOURS` lists every directed offset used for *force accumulation* in the
 * solver (each node pulls toward all of its existing neighbours). `RENDER_SPRINGS`
 * lists the positive-direction subset used to draw each spring exactly once.
 */

export type PinMode = 'top-row' | 'top-corners' | 'none';

export interface SpringOffset {
  dx: number;
  dy: number;
  /** Stiffness multiplier relative to the global spring constant. */
  k: number;
  /** Rest length as a multiple of the grid spacing. */
  rest: number;
}

const SQRT2 = Math.SQRT2;

export const NEIGHBOURS: readonly SpringOffset[] = [
  // structural
  { dx: 1, dy: 0, k: 1.0, rest: 1 },
  { dx: -1, dy: 0, k: 1.0, rest: 1 },
  { dx: 0, dy: 1, k: 1.0, rest: 1 },
  { dx: 0, dy: -1, k: 1.0, rest: 1 },
  // shear
  { dx: 1, dy: 1, k: 0.7, rest: SQRT2 },
  { dx: -1, dy: 1, k: 0.7, rest: SQRT2 },
  { dx: 1, dy: -1, k: 0.7, rest: SQRT2 },
  { dx: -1, dy: -1, k: 0.7, rest: SQRT2 },
  // bend
  { dx: 2, dy: 0, k: 0.25, rest: 2 },
  { dx: -2, dy: 0, k: 0.25, rest: 2 },
  { dx: 0, dy: 2, k: 0.25, rest: 2 },
  { dx: 0, dy: -2, k: 0.25, rest: 2 },
];

/** Positive-direction representatives, so each undirected spring is drawn once. */
export const RENDER_SPRINGS: readonly SpringOffset[] = [
  { dx: 1, dy: 0, k: 1.0, rest: 1 },
  { dx: 0, dy: 1, k: 1.0, rest: 1 },
  { dx: 1, dy: 1, k: 0.7, rest: SQRT2 },
  { dx: 1, dy: -1, k: 0.7, rest: SQRT2 },
  { dx: 2, dy: 0, k: 0.25, rest: 2 },
  { dx: 0, dy: 2, k: 0.25, rest: 2 },
];

export interface Topology {
  cols: number;
  rows: number;
  count: number;
  spacing: number;
  /** Rest positions, count * 3 (xyz), centred on the origin. */
  home: Float32Array;
  /** 1 = pinned, 0 = free; length `count`. */
  fixed: Float32Array;
  /** Endpoint index pairs (a, b) for each drawn spring; length springCount * 2. */
  springs: Uint32Array;
  /** Rest length per drawn spring; length springCount. */
  springRest: Float32Array;
  springCount: number;
}

/** A tiny deterministic hash → [0,1), used to seed out-of-plane buckling. */
function hash(i: number): number {
  const s = Math.sin(i * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

export function buildTopology(
  cols: number,
  rows: number,
  spacing: number,
  pin: PinMode,
): Topology {
  const count = cols * rows;
  const home = new Float32Array(count * 3);
  const fixed = new Float32Array(count);

  const cx = (cols - 1) / 2;
  const cy = (rows - 1) / 2;

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      home[i * 3 + 0] = (x - cx) * spacing;
      home[i * 3 + 1] = (cy - y) * spacing; // y = 0 is the top row
      // Small out-of-plane seed so a pinned sheet buckles into 3D folds
      // instead of staying perfectly planar under gravity.
      home[i * 3 + 2] = (hash(i) - 0.5) * spacing * 0.6;

      let pinned = false;
      if (pin === 'top-row') pinned = y === 0;
      else if (pin === 'top-corners') pinned = y === 0 && (x === 0 || x === cols - 1);
      fixed[i] = pinned ? 1 : 0;
    }
  }

  const a: number[] = [];
  const rest: number[] = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = y * cols + x;
      for (const o of RENDER_SPRINGS) {
        const nx = x + o.dx;
        const ny = y + o.dy;
        if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
        const j = ny * cols + nx;
        a.push(i, j);
        rest.push(o.rest * spacing);
      }
    }
  }

  return {
    cols,
    rows,
    count,
    spacing,
    home,
    fixed,
    springs: Uint32Array.from(a),
    springRest: Float32Array.from(rest),
    springCount: rest.length,
  };
}
