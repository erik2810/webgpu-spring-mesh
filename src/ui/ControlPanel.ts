import type { SimParams } from '../core/params';
import type { PinMode } from '../core/topology';

export interface PanelStats {
  fps: number;
  nodes: number;
  springs: number;
}

export interface PanelCallbacks {
  /** Fired on any parameter change. `structural` ⇒ the mesh must be rebuilt. */
  onParam(params: SimParams, structural: boolean): void;
  onReset(): void;
  onPauseToggle(paused: boolean): void;
}

interface SliderSpec {
  key: keyof SimParams;
  name: string;
  min: number;
  max: number;
  step: number;
  structural?: boolean;
  format: (v: number) => string;
}

const ICONS = {
  reset:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7M3 4v4h4"/></svg>',
  pause: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>',
  play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4l13 8-13 8z"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
  hand: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 9l-2 2 4 7h9l3-7-2-1-2 2V5a1.5 1.5 0 0 0-3 0v4M11 9V4a1.5 1.5 0 0 0-3 0v6M14 9V6"/></svg>',
};

let uid = 0;
const nextId = (): string => `ctl-${++uid}`;

/** Glassy control overlay built from plain DOM (no UI framework). */
export class ControlPanel {
  private readonly root: HTMLElement;
  private readonly hint: HTMLElement;
  private readonly statEls: Record<string, HTMLElement> = {};
  private paused = false;
  private collapsed = false;

  constructor(
    parent: HTMLElement,
    private readonly params: SimParams,
    backendLabel: string,
    maxDensity: number,
    private readonly cb: PanelCallbacks,
  ) {
    this.root = el('aside', 'panel');
    this.root.setAttribute('aria-label', 'Simulation controls');

    // header
    const head = el('div', 'panel__head');
    head.append(el('span', 'panel__dot'));
    const titles = el('div');
    const h = el('h1', 'panel__title', 'Spring Mesh');
    const sub = el('p', 'panel__sub', backendLabel);
    titles.append(h, sub);
    const collapse = iconButton('panel__collapse btn btn--icon', ICONS.chevron, 'Collapse panel');
    collapse.setAttribute('aria-expanded', 'true');
    collapse.addEventListener('click', () => this.toggleCollapse(collapse));
    head.append(titles, collapse);

    const body = el('div', 'panel__body');

    // --- physics ---
    const physics: SliderSpec[] = [
      { key: 'gravity', name: 'Gravity', min: 0, max: 25, step: 0.1, format: (v) => v.toFixed(1) },
      { key: 'stiffness', name: 'Stiffness', min: 0, max: 1, step: 0.02, format: (v) => v.toFixed(2) },
      {
        key: 'damping',
        name: 'Damping',
        min: 0.9,
        max: 1,
        step: 0.001,
        format: (v) => v.toFixed(3),
      },
      { key: 'mass', name: 'Mass', min: 0.2, max: 4, step: 0.1, format: (v) => v.toFixed(1) },
    ];
    body.append(this.section('Physics', physics.map((s) => this.slider(s))));

    // --- wind ---
    const wind: SliderSpec[] = [
      {
        key: 'windStrength',
        name: 'Wind strength',
        min: 0,
        max: 40,
        step: 0.5,
        format: (v) => v.toFixed(1),
      },
      {
        key: 'windFreq',
        name: 'Wind frequency',
        min: 0.2,
        max: 8,
        step: 0.1,
        format: (v) => v.toFixed(1),
      },
    ];
    body.append(this.section('Wind', wind.map((s) => this.slider(s))));

    // --- topology ---
    const densitySlider = this.slider({
      key: 'density',
      name: 'Topology density',
      min: 12,
      max: maxDensity,
      step: 1,
      structural: true,
      format: (v) => `${v}×${v}`,
    });
    const pin = this.segmented<PinMode>(
      'Pinning',
      [
        ['top-corners', 'Corners'],
        ['top-row', 'Top edge'],
        ['none', 'Free'],
      ],
      this.params.pin,
      (v) => {
        this.params.pin = v;
        this.cb.onParam(this.params, true);
      },
    );
    const floor = this.toggle('Floor collision', this.params.floorOn, (v) => {
      this.params.floorOn = v;
      this.cb.onParam(this.params, false);
    });
    body.append(this.section('Topology', [densitySlider, pin, floor]));

    // --- simulation ---
    const substeps = this.segmented<number>(
      'Substeps / frame',
      [
        [2, '2'],
        [4, '4'],
        [6, '6'],
        [8, '8'],
      ],
      this.params.substeps,
      (v) => {
        this.params.substeps = v;
        this.cb.onParam(this.params, false);
      },
    );
    const resetBtn = button('btn', `${ICONS.reset}<span>Reset</span>`, 'Reset simulation');
    resetBtn.addEventListener('click', () => this.cb.onReset());
    const pauseBtn = button('btn', `${ICONS.pause}<span>Pause</span>`, 'Pause simulation');
    pauseBtn.addEventListener('click', () => {
      this.paused = !this.paused;
      pauseBtn.innerHTML = this.paused
        ? `${ICONS.play}<span>Resume</span>`
        : `${ICONS.pause}<span>Pause</span>`;
      pauseBtn.setAttribute('aria-label', this.paused ? 'Resume simulation' : 'Pause simulation');
      this.cb.onPauseToggle(this.paused);
    });
    const btnRow = el('div', 'btn-row');
    btnRow.append(resetBtn, pauseBtn);
    body.append(this.section('Simulation', [substeps, btnRow]));

    // --- stats ---
    body.append(this.statsSection());

    this.root.append(head, body);
    parent.append(this.root);

    // bottom-left hint pill
    this.hint = el('div', 'hint');
    this.hint.innerHTML = `${ICONS.hand}<span><b>Drag a node</b> to disrupt · drag background to orbit · scroll to zoom</span>`;
    parent.append(this.hint);
  }

  private section(label: string, children: HTMLElement[]): HTMLElement {
    const s = el('section', 'section');
    s.append(el('div', 'section__label', label), ...children);
    return s;
  }

  private slider(spec: SliderSpec): HTMLElement {
    const id = nextId();
    const wrap = el('label', 'ctl');
    wrap.setAttribute('for', id);
    const row = el('div', 'ctl__row');
    const name = el('span', 'ctl__name', spec.name);
    const val = el('span', 'ctl__val', spec.format(this.params[spec.key] as number));
    row.append(name, val);

    const input = document.createElement('input');
    input.type = 'range';
    input.id = id;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(this.params[spec.key] as number);
    input.setAttribute('aria-label', spec.name);
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      (this.params[spec.key] as number) = v;
      val.textContent = spec.format(v);
      this.cb.onParam(this.params, !!spec.structural);
    });

    wrap.append(row, input);
    return wrap;
  }

  private segmented<T extends string | number>(
    label: string,
    options: [T, string][],
    current: T,
    onSelect: (v: T) => void,
  ): HTMLElement {
    const wrap = el('div', 'ctl');
    wrap.append(el('div', 'ctl__name', label));
    const group = el('div', 'seg');
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', label);
    const btns: HTMLButtonElement[] = [];
    for (const [value, text] of options) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'seg__btn';
      b.textContent = text;
      b.setAttribute('aria-pressed', String(value === current));
      b.addEventListener('click', () => {
        btns.forEach((x) => x.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        onSelect(value);
      });
      btns.push(b);
      group.append(b);
    }
    wrap.append(group);
    return wrap;
  }

  private toggle(label: string, on: boolean, onChange: (v: boolean) => void): HTMLElement {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'toggle';
    b.setAttribute('aria-pressed', String(on));
    b.innerHTML = `<span class="toggle__name">${label}</span><span class="switch" aria-hidden="true"></span>`;
    b.addEventListener('click', () => {
      const next = b.getAttribute('aria-pressed') !== 'true';
      b.setAttribute('aria-pressed', String(next));
      onChange(next);
    });
    return b;
  }

  private statsSection(): HTMLElement {
    const s = el('section', 'section');
    const grid = el('div', 'stats');
    const make = (key: string, label: string): void => {
      const cell = el('div', 'stat');
      cell.append(el('div', 'stat__k', label));
      const v = el('div', 'stat__v', '—');
      this.statEls[key] = v;
      cell.append(v);
      grid.append(cell);
    };
    make('fps', 'FPS');
    make('nodes', 'Nodes');
    make('springs', 'Springs');
    make('backend', 'Solver');
    s.append(grid);
    return s;
  }

  private toggleCollapse(btn: HTMLElement): void {
    this.collapsed = !this.collapsed;
    this.root.classList.toggle('is-collapsed', this.collapsed);
    btn.setAttribute('aria-expanded', String(!this.collapsed));
    btn.setAttribute('aria-label', this.collapsed ? 'Expand panel' : 'Collapse panel');
    (btn.querySelector('svg') as SVGElement)?.style.setProperty(
      'transform',
      this.collapsed ? 'rotate(-90deg)' : 'none',
    );
  }

  setStats(stats: PanelStats): void {
    const fps = this.statEls.fps;
    fps.textContent = String(Math.round(stats.fps));
    fps.className = `stat__v ${stats.fps >= 55 ? 'is-good' : stats.fps < 30 ? 'is-warn' : ''}`;
    this.statEls.nodes.textContent = stats.nodes.toLocaleString();
    this.statEls.springs.textContent = stats.springs.toLocaleString();
  }

  setBackend(label: string): void {
    this.statEls.backend.textContent = label;
  }

  destroy(): void {
    this.root.remove();
    this.hint.remove();
  }
}

// --- tiny DOM helpers --------------------------------------------------------
function el(tag: string, className = '', text = ''): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(className: string, html: string, ariaLabel: string): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = className;
  b.innerHTML = html;
  b.setAttribute('aria-label', ariaLabel);
  return b;
}

function iconButton(className: string, html: string, ariaLabel: string): HTMLButtonElement {
  return button(className, html, ariaLabel);
}
