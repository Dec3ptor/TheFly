// three.js scene for the connectome viewer: translucent neuropil shells for context,
// one LineSegments object per loaded neuron, and click-to-select picking.

import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { heatColor } from './sim.js';

// Source data is in nanometres and the CNS is about a millimetre long. We render in
// micrometres so camera distances, raycaster thresholds and depth precision all sit in
// a comfortable numeric range.
const NM_TO_UM = 0.001;

// Centre of the full CNS bounding box, in nanometres, measured from the shell meshes.
const CNS_CENTRE_NM = [384256, 306911, 578048];

const PALETTE = [
  0x4cc9f0, 0xf72585, 0x4895ef, 0xffb703, 0x80ed99, 0xb5179e,
  0x00d4a6, 0xff7b00, 0x7b6cf6, 0xf9c74f, 0x43aa8b, 0xff5d8f,
];

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.neurons = new Map();   // bodyId -> { object, record, color }
    this.shells = [];
    this.onSelect = null;
    this.colorCursor = 0;
    this.activityMode = false;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setClearColor(0x080b14, 1);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(38, 1, 1, 20000);
    // The CNS runs along Z with the brain at low Z, so -Z is anatomical "up".
    this.camera.up.set(0, 0, -1);
    this.camera.position.set(0, -1500, 0);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.7;

    // Everything is added under this group, which converts nanometres to centred
    // micrometres so individual geometries can keep their source coordinates.
    this.root = new THREE.Group();
    this.root.scale.setScalar(NM_TO_UM);
    this.root.position.set(
      -CNS_CENTRE_NM[0] * NM_TO_UM,
      -CNS_CENTRE_NM[1] * NM_TO_UM,
      -CNS_CENTRE_NM[2] * NM_TO_UM,
    );
    this.scene.add(this.root);

    this.raycaster = new THREE.Raycaster();
    // Skeletons are infinitely thin lines, so picking needs a tolerance in world
    // units (micrometres) or they are essentially unclickable.
    this.raycaster.params.Line.threshold = 3;

    canvas.addEventListener('pointerdown', (e) => this._onPointerDown(e));
    canvas.addEventListener('pointerup', (e) => this._onPointerUp(e));

    this._resize();
    window.addEventListener('resize', () => this._resize());
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  _resize() {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (!w || !h) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Add a neuropil shell as a ghostly additive surface behind the neurons. */
  addShell({ positions, indices }, { color = 0x2a4a7f, opacity = 0.19 } = {}) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      // Drawing only the far wall of the shell halves the overdraw and keeps the
      // near surface from washing out the neurons sitting inside it.
      side: THREE.BackSide,
      depthWrite: false,          // keep the shell from occluding neurons inside it
      blending: THREE.AdditiveBlending,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = -1;
    mesh.raycast = () => {};      // context geometry is never a click target
    this.root.add(mesh);
    this.shells.push(mesh);
    return mesh;
  }

  setShellsVisible(visible) {
    for (const mesh of this.shells) mesh.visible = visible;
  }

  nextColor() {
    return PALETTE[this.colorCursor++ % PALETTE.length];
  }

  addNeuron(record, skeleton, color = this.nextColor()) {
    if (this.neurons.has(record.bodyId)) return this.neurons.get(record.bodyId);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(skeleton.segments, 3));

    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 });
    const object = new THREE.LineSegments(geometry, material);
    object.userData.bodyId = record.bodyId;

    this.root.add(object);
    const entry = { object, record, color };
    this.neurons.set(record.bodyId, entry);
    return entry;
  }

  removeNeuron(bodyId) {
    const entry = this.neurons.get(bodyId);
    if (!entry) return;
    this.root.remove(entry.object);
    entry.object.geometry.dispose();
    entry.object.material.dispose();
    this.neurons.delete(bodyId);
  }

  clearNeurons() {
    for (const bodyId of [...this.neurons.keys()]) this.removeNeuron(bodyId);
    this.colorCursor = 0;
  }

  /** Dim every neuron except one, or restore all of them when given null. */
  highlight(bodyId) {
    this.selectedBody = bodyId;
    if (this.activityMode) return;      // activity colouring owns the palette
    for (const [id, entry] of this.neurons) {
      const isTarget = bodyId === null || id === bodyId;
      entry.object.material.opacity = isTarget ? 0.95 : 0.12;
      entry.object.material.color.setHex(isTarget ? entry.color : 0x5a6478);
    }
  }

  /**
   * Colour every neuron by the activity of the cell type it belongs to. Activity is
   * per type, not per neuron, because that is the resolution the model runs at.
   */
  paintActivity(rate) {
    this.activityMode = true;
    for (const entry of this.neurons.values()) {
      const activity = rate[entry.record.typeIndex] || 0;
      entry.object.material.color.setHex(heatColor(activity));
      // Idle neurons stay faintly visible so the anatomy does not disappear.
      entry.object.material.opacity = 0.18 + 0.8 * activity;
    }
  }

  /** Leave activity colouring and go back to one colour per neuron. */
  clearActivity() {
    this.activityMode = false;
    this.highlight(this.selectedBody ?? null);
  }

  _onPointerDown(event) {
    this._pressed = { x: event.clientX, y: event.clientY };
  }

  _onPointerUp(event) {
    // Ignore the pointerup that ends a camera drag; only treat a near-stationary
    // click as a selection attempt.
    if (!this._pressed) return;
    const moved = Math.hypot(event.clientX - this._pressed.x, event.clientY - this._pressed.y);
    this._pressed = null;
    if (moved > 4 || !this.onSelect) return;

    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);

    const targets = [...this.neurons.values()].map((entry) => entry.object);
    const hit = this.raycaster.intersectObjects(targets, false)[0];
    this.onSelect(hit ? hit.object.userData.bodyId : null);
  }

  /** Frame the loaded neurons, or the whole CNS when nothing is loaded. */
  frame() {
    const box = new THREE.Box3();
    for (const entry of this.neurons.values()) box.expandByObject(entry.object);
    if (box.isEmpty()) for (const mesh of this.shells) box.expandByObject(mesh);
    if (box.isEmpty()) return;

    const centre = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 20);
    const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);

    this.controls.target.copy(centre);
    // Approach along the existing view direction so framing does not throw away
    // the orientation the user has rotated to.
    const direction = this.camera.position.clone().sub(this.controls.target);
    if (direction.lengthSq() < 1e-6) direction.set(0, -1, 0);
    this.camera.position.copy(centre).add(direction.normalize().multiplyScalar(distance * 1.1));
    this.camera.near = Math.max(distance / 1000, 0.5);
    this.camera.far = distance * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resetView() {
    this.controls.target.set(0, 0, 0);
    this.camera.position.set(0, -1500, 0);
    this.camera.up.set(0, 0, -1);
    this.frame();
  }
}
