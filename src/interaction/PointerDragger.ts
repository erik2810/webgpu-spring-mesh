import * as THREE from 'three/webgpu';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import type { ClothSim } from '../core/types';

/**
 * Click-and-drag a node to disrupt the mesh.
 *
 * On press we take one position snapshot, pick the node nearest the pointer ray,
 * and pin it. While dragging we project the pointer onto a camera-facing plane
 * through that node and feed the result to {@link ClothSim.setGrab}. OrbitControls
 * is suspended for the duration so a drag never spins the camera.
 */
export class PointerDragger {
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly plane = new THREE.Plane();
  private readonly hit = new THREE.Vector3();
  private readonly tmp = new THREE.Vector3();

  private grabbing = false;
  private grabIndex = -1;
  private busy = false;

  constructor(
    private readonly dom: HTMLElement,
    private readonly camera: THREE.PerspectiveCamera,
    private readonly sim: ClothSim,
    private readonly controls: OrbitControls,
  ) {
    this.dom.addEventListener('pointerdown', this.onDown);
    this.dom.addEventListener('pointermove', this.onMove);
    window.addEventListener('pointerup', this.onUp);
    window.addEventListener('pointercancel', this.onUp);
  }

  private setPointer(e: PointerEvent): void {
    const r = this.dom.getBoundingClientRect();
    this.pointer.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  private onDown = async (e: PointerEvent): Promise<void> => {
    if (e.button !== 0 || this.busy) return;
    this.busy = true;
    this.setPointer(e);

    const positions = await this.sim.snapshotPositions();
    const ray = this.raycaster.ray;
    const pickR = this.sim.spacing * 1.6;
    const pickR2 = pickR * pickR;

    let best = -1;
    let bestPerp = pickR2;
    let bestDepth = Infinity;
    for (let i = 0; i < this.sim.count; i++) {
      this.tmp.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      const perp = ray.distanceSqToPoint(this.tmp);
      if (perp > pickR2) continue;
      const depth = ray.origin.distanceToSquared(this.tmp);
      // Prefer the node closest to the ray; break near-ties by nearest to camera.
      if (perp < bestPerp - 1e-6 || (Math.abs(perp - bestPerp) <= 1e-6 && depth < bestDepth)) {
        best = i;
        bestPerp = perp;
        bestDepth = depth;
      }
    }

    this.busy = false;
    if (best < 0) return;

    this.tmp.set(positions[best * 3], positions[best * 3 + 1], positions[best * 3 + 2]);
    this.camera.getWorldDirection(this.plane.normal);
    this.plane.setFromNormalAndCoplanarPoint(this.plane.normal, this.tmp);

    this.grabbing = true;
    this.grabIndex = best;
    this.controls.enabled = false;
    this.dom.style.cursor = 'grabbing';
    this.dom.setPointerCapture?.(e.pointerId);
    this.sim.setGrab(best, this.tmp);
  };

  private onMove = (e: PointerEvent): void => {
    if (!this.grabbing) return;
    this.setPointer(e);
    if (this.raycaster.ray.intersectPlane(this.plane, this.hit)) {
      this.sim.setGrab(this.grabIndex, this.hit);
    }
  };

  private onUp = (e: PointerEvent): void => {
    if (!this.grabbing) return;
    this.grabbing = false;
    this.grabIndex = -1;
    this.sim.clearGrab();
    this.controls.enabled = true;
    this.dom.style.cursor = 'grab';
    this.dom.releasePointerCapture?.(e.pointerId);
  };

  dispose(): void {
    this.dom.removeEventListener('pointerdown', this.onDown);
    this.dom.removeEventListener('pointermove', this.onMove);
    window.removeEventListener('pointerup', this.onUp);
    window.removeEventListener('pointercancel', this.onUp);
  }
}
