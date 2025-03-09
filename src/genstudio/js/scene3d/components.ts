// components.ts

import {
  billboardVertCode,
  billboardFragCode,
  billboardPickingVertCode,
  ellipsoidVertCode,
  ellipsoidFragCode,
  ellipsoidPickingVertCode,
  ringVertCode,
  ringFragCode,
  ringPickingVertCode,
  cuboidVertCode,
  cuboidFragCode,
  cuboidPickingVertCode,
  lineBeamVertCode,
  lineBeamFragCode,
  lineBeamPickingVertCode,
  pickingFragCode,
  POINT_CLOUD_GEOMETRY_LAYOUT,
  POINT_CLOUD_INSTANCE_LAYOUT,
  POINT_CLOUD_PICKING_INSTANCE_LAYOUT,
  MESH_GEOMETRY_LAYOUT,
  ELLIPSOID_INSTANCE_LAYOUT,
  ELLIPSOID_PICKING_INSTANCE_LAYOUT,
  LINE_BEAM_INSTANCE_LAYOUT,
  LINE_BEAM_PICKING_INSTANCE_LAYOUT,
  CUBOID_INSTANCE_LAYOUT,
  CUBOID_PICKING_INSTANCE_LAYOUT,
  RING_INSTANCE_LAYOUT,
  RING_PICKING_INSTANCE_LAYOUT,
} from "./shaders";

import {
  createCubeGeometry,
  createBeamGeometry,
  createSphereGeometry,
  createTorusGeometry,
} from "./geometry";

import { packID } from "./picking";

import {
  BaseComponentConfig,
  Decoration,
  PipelineCacheEntry,
  PrimitiveSpec,
  PipelineConfig,
  GeometryResource,
  GeometryData,
  ElementConstants,
} from "./types";

/** ===================== DECORATIONS + COMMON UTILS ===================== **/

/** Helper function to apply decorations to an array of instances */
function applyDecorations(
  decorations: Decoration[] | undefined,
  instanceCount: number,
  setter: (i: number, dec: Decoration) => void
) {
  if (!decorations) return;
  for (const dec of decorations) {
    if (!dec.indexes) continue;
    for (const idx of dec.indexes) {
      if (idx < 0 || idx >= instanceCount) continue;
      setter(idx, dec);
    }
  }
}

/** Helper function to handle sorted indices and position mapping */
function getIndicesAndMapping(
  count: number,
  sortedIndices?: Uint32Array
): {
  indices: Uint32Array | null; // Change to Uint32Array
  indexToPosition: Uint32Array | null;
} {
  if (!sortedIndices) {
    return {
      indices: null,
      indexToPosition: null,
    };
  }

  // Only create mapping if we have sorted indices
  const indexToPosition = new Uint32Array(count);
  for (let j = 0; j < count; j++) {
    indexToPosition[sortedIndices[j]] = j;
  }

  return {
    indices: sortedIndices,
    indexToPosition,
  };
}


function acopy(source: ArrayLike<number>, sourceI: number, out: ArrayLike<number> & { [n: number]: number }, outI: number, n: number) {
  for (let i = 0; i < n; i++) {
    out[outI + i] = source[sourceI + i];
  }
}

/** ===================== MINI-FRAMEWORK FOR RENDER/PICK DATA ===================== **/

function applyDefaultDecoration(
  out: Float32Array,
  offset: number,
  dec: Decoration,
  spec: PrimitiveSpec<any>
) {
  if (dec.color) {
    out[offset + spec.colorOffset + 0] = dec.color[0];
    out[offset + spec.colorOffset + 1] = dec.color[1];
    out[offset + spec.colorOffset + 2] = dec.color[2];
  }
  if (dec.alpha !== undefined) {
    out[offset + spec.alphaOffset] = dec.alpha;
  }
  if (dec.scale !== undefined) {
    spec.applyDecorationScale(out, offset, dec.scale);
  }
}

/**
 * Builds render data for any shape using the shape's fillRenderGeometry callback
 * plus the standard columnar/default color and alpha usage, sorted index handling,
 * and decoration loop.
 */
export function buildRenderData<ConfigType extends BaseComponentConfig>(
  elem: ConfigType,
  spec: PrimitiveSpec<ConfigType>,
  out: Float32Array,
  sortedIndices?: Uint32Array
): boolean {
  const count = spec.getCount(elem);
  if (count === 0) return false;

  // Retrieve base defaults (color, alpha)
  const constants = getElementConstants(spec, elem);

  const { indices, indexToPosition } = getIndicesAndMapping(
    count,
    sortedIndices
  );
  const floatsPerInstance = spec.getFloatsPerInstance();

  for (let j = 0; j < count; j++) {
    const i = indices ? indices[j] : j;
    const offset = j * floatsPerInstance;

    // Let the shape fill the geometry portion (positions, sizes, quaternions, etc.)
    spec.fillRenderGeometry(elem, i, out, offset);

    // Color / alpha usage is handled here
    const colorIndex = spec.getColorIndexForInstance
      ? spec.getColorIndexForInstance(elem, i)
      : i;
    if (constants.color) {
      acopy(constants.color, 0, out, offset + spec.colorOffset, 3);
    } else {
      acopy(elem.colors!, colorIndex * 3, out, offset + spec.colorOffset, 3);
    }
    out[offset + spec.alphaOffset] = constants.alpha || elem.alphas![colorIndex];
  }

  applyDecorations(elem.decorations, count, (idx, dec) => {
    const j = indexToPosition ? indexToPosition[idx] : idx;
    if (j < 0 || j >= count) return;

    if (spec.applyDecoration) {
      // Use component-specific decoration handling
      spec.applyDecoration(out, j, dec, floatsPerInstance);
    } else {
      applyDefaultDecoration(out, j * floatsPerInstance, dec, spec);
    }
  });

  return true;
}

/**
 * Builds picking data for any shape using the shape's fillPickingGeometry callback,
 * plus handling sorted indices, decorations that affect scale, and base pick ID.
 */
export function buildPickingData<ConfigType extends BaseComponentConfig>(
  elem: ConfigType,
  spec: PrimitiveSpec<ConfigType>,
  out: Float32Array,
  baseID: number,
  sortedIndices?: Uint32Array
): void {
  const count = spec.getCount(elem);
  if (count === 0) return;

  const { indices, indexToPosition } = getIndicesAndMapping(
    count,
    sortedIndices
  );
  const floatsPerPicking = spec.getFloatsPerPicking();

  // Do the main fill
  for (let j = 0; j < count; j++) {
    const i = indices ? indices[j] : j;
    const offset = j * floatsPerPicking;
    // Let the shape fill the picking geometry (positions, orientation, pickID)
    spec.fillPickingGeometry(elem, i, out, offset, baseID); // scale=1.0 initially
  }

  // Then apply decorations that affect scale
  applyDecorations(elem.decorations, count, (idx, dec) => {
    if (dec.scale === undefined || !spec.applyDecorationScale) return;
    const j = indexToPosition ? indexToPosition[idx] : idx;
    if (j < 0 || j >= count) return;

    if (spec.applyDecoration) {
      spec.applyDecoration(out, j, dec, floatsPerPicking);
    } else {
      spec.applyDecorationScale(out, j * floatsPerPicking, dec.scale);
    }
  });
}

/** ===================== GPU PIPELINE HELPERS (unchanged) ===================== **/

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  createFn: () => GPURenderPipeline,
  cache: Map<string, PipelineCacheEntry> // This will be the instance cache
): GPURenderPipeline {
  const entry = cache.get(key);
  if (entry && entry.device === device) {
    return entry.pipeline;
  }

  // Create new pipeline and cache it with device reference
  const pipeline = createFn();
  cache.set(key, { pipeline, device });
  return pipeline;
}

function createRenderPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat
): GPURenderPipeline {
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // Get primitive configuration with defaults
  const primitiveConfig = {
    topology: config.primitive?.topology || "triangle-list",
    cullMode: config.primitive?.cullMode || "back",
  };

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: config.vertexShader }),
      entryPoint: config.vertexEntryPoint,
      buffers: config.bufferLayouts,
    },
    fragment: {
      module: device.createShaderModule({ code: config.fragmentShader }),
      entryPoint: config.fragmentEntryPoint,
      targets: [
        {
          format,
          writeMask: config.colorWriteMask ?? GPUColorWrite.ALL,
          ...(config.blend && {
            blend: {
              color: config.blend.color || {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: config.blend.alpha || {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
            },
          }),
        },
      ],
    },
    primitive: primitiveConfig,
    depthStencil: config.depthStencil || {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });
}

function createTranslucentGeometryPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any> // Take the primitive spec instead of just type
): GPURenderPipeline {
  return createRenderPipeline(
    device,
    bindGroupLayout,
    {
      ...config,
      primitive: primitiveSpec.renderConfig,
      blend: {
        color: {
          srcFactor: "src-alpha",
          dstFactor: "one-minus-src-alpha",
          operation: "add",
        },
        alpha: {
          srcFactor: "one",
          dstFactor: "one-minus-src-alpha",
          operation: "add",
        },
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    },
    format
  );
}

const createBuffers = (
  device: GPUDevice,
  { vertexData, indexData }: GeometryData
): GeometryResource => {
  const vb = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb, 0, vertexData);

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ib, 0, indexData);

  // Each vertex has 6 floats (position + normal)
  const vertexCount = vertexData.length / 6;

  return {
    vb,
    ib,
    indexCount: indexData.length,
    vertexCount,
  };
};

const computeConstants = (spec: any, elem: any) => {
  const constants: ElementConstants = {};

  for (const [key, defaultValue] of Object.entries({
    alpha: 1.0,
    color: [0.5, 0.5, 0.5],
    ...spec.defaults,
  })) {
    const pluralKey = key + "s";
    const pluralValue = elem[pluralKey];
    const singularValue = elem[key];

    const targetTypeIsArray = Array.isArray(defaultValue);

    // Case 1: No plural form exists. Use element value or default.
    if (!pluralValue) {
      if (targetTypeIsArray && typeof singularValue === 'number') {
        // Fill array with the single number value
        // @ts-ignore
        constants[key as keyof ElementConstants] = new Array(defaultValue.length).fill(singularValue);
      } else {
        constants[key as keyof ElementConstants] = singularValue || defaultValue;
      }
      continue;
    }
    // Case 2: Target value is an array, and the specified plural is of that length, so use it as a constant value.
    if (targetTypeIsArray && pluralValue.length === defaultValue.length) {
      constants[key as keyof ElementConstants] = pluralValue || defaultValue;
      continue;
    }

    // Case 3: Target value is an array, and the specified plural is of length 1, repeat it.
    if (targetTypeIsArray && pluralValue.length === 1) {
      // Fill array with the single value
      const filledArray = new Array((defaultValue as number[]).length).fill(
        pluralValue[0]
      );
      // @ts-ignore
      constants[key as keyof ElementConstants] = filledArray;
    }
  }

  return constants;
};

const getElementConstants = (
  spec: PrimitiveSpec<BaseComponentConfig>,
  elem: BaseComponentConfig
): ElementConstants => {
  if (elem.constants) return elem.constants;
  elem.constants = computeConstants(spec, elem);
  return elem.constants;
};

/** ===================== POINT CLOUD ===================== **/

export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: "PointCloud";
  centers: Float32Array;
  sizes?: Float32Array; // Per-point sizes
  size?: number; // Default size, defaults to 0.02
}

export const pointCloudSpec: PrimitiveSpec<PointCloudComponentConfig> = {
  type: "PointCloud",

  defaults: {
    size: 0.02,
  },

  getCount(elem) {
    return elem.centers.length / 3;
  },

  getFloatsPerInstance() {
    return 8; // position(3) + size(1) + color(3) + alpha(1) = 8
  },

  getFloatsPerPicking() {
    return 5; // position(3) + size(1) + pickID(1) = 5
  },

  getCenters(elem) {
    return elem.centers;
  },

  // Geometry Offsets
  colorOffset: 4, // color starts at out[offset+4]
  alphaOffset: 7, // alpha is at out[offset+7]

  // fillRenderGeometry: shape-specific code, ignoring color/alpha
  fillRenderGeometry(elem, i, out, offset) {
    const constants = getElementConstants(this, elem);

    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Size - use constant or per-instance value
    out[offset + 3] = constants.size || elem.sizes![i];

    // Color - use constant or per-instance value
    if (constants.color) {
      acopy(constants.color, 0, out, offset + 4, 3);
    } else {
      acopy(elem.colors!, i * 3, out, offset + 4, 3);
    }

    // Alpha - use constant or per-instance value
    out[offset + 7] = constants.alpha ?? elem.alphas![i];
  },

  // For decorations that scale the point size
  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
  },

  // fillPickingGeometry
  fillPickingGeometry(elem, i, out, offset, baseID) {
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    const constants = getElementConstants(this, elem);
    const pointSize = constants.size || elem.sizes![i];
    out[offset + 3] = pointSize;

    // pickID
    out[offset + 4] = packID(baseID + i);
  },
  // Rendering configuration
  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  // Pipeline creation methods
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "PointCloudShading",
      () =>
        createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: billboardVertCode,
            fragmentShader: billboardFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [
              POINT_CLOUD_GEOMETRY_LAYOUT,
              POINT_CLOUD_INSTANCE_LAYOUT,
            ],
            primitive: this.renderConfig,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
            depthStencil: {
              format: "depth24plus",
              depthWriteEnabled: true,
              depthCompare: "less",
            },
          },
          format
        ),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "PointCloudPicking",
      () =>
        createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: billboardPickingVertCode,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [
              POINT_CLOUD_GEOMETRY_LAYOUT,
              POINT_CLOUD_PICKING_INSTANCE_LAYOUT,
            ],
          },
          "rgba8unorm"
        ),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, {
      vertexData: new Float32Array([
        -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 1.0, -0.5,
        0.5, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0,
      ]),
      indexData: new Uint16Array([0, 1, 2, 2, 1, 3]),
    });
  },
};

/** ===================== ELLIPSOID ===================== **/

export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: 'Ellipsoid' | 'EllipsoidAxes';
  centers: Float32Array | number[];
  half_sizes?: Float32Array | number[];
  half_size?: [number, number, number] | number;
  quaternions?: Float32Array | number[];
  quaternion?: [number, number, number, number];
  fill_mode?: 'Solid' | 'MajorWireframe';
}

export const ellipsoidSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  type: "Ellipsoid",

  defaults: {
    half_size: [0.5, 0.5, 0.5],
    quaternion: [0, 0, 0, 1],
  },

  getCount(elem) {
    return elem.centers.length / 3;
  },

  getFloatsPerInstance() {
    // pos(3) + size(3) + quat(4) + color(3) + alpha(1) = 14
    return 14;
  },

  getFloatsPerPicking() {
    // pos(3) + size(3) + quat(4) + pickID(1) = 11
    return 11;
  },

  getCenters(elem) {
    return elem.centers;
  },

  // Where color/alpha go
  colorOffset: 10,
  alphaOffset: 13,

  fillRenderGeometry(elem, i, out, offset) {
    const constants = getElementConstants(this, elem);

    // Position
    acopy(elem.centers, i * 3, out, offset, 3)


    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3)
    } else {
      acopy(elem.half_sizes as ArrayLike<number>, i * 3, out, offset + 3, 3)
    }

    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4)
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4)
    }
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // Multiply the sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, i, out, offset, baseID) {
    const constants = getElementConstants(this, elem);

    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3);
    } else {
      acopy(elem.half_sizes as ArrayLike<number>, i * 3, out, offset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4);
    }

    // picking ID
    out[offset + 10] = packID(baseID + i);
  },

  renderConfig: {
    cullMode: "back",
    topology: "triangle-list",
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidShading",
      () => {
        return createTranslucentGeometryPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ellipsoidVertCode,
            fragmentShader: ellipsoidFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT],
          },
          format,
          ellipsoidSpec
        );
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidPicking",
      () => {
        return createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ellipsoidPickingVertCode,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [
              MESH_GEOMETRY_LAYOUT,
              ELLIPSOID_PICKING_INSTANCE_LAYOUT,
            ],
          },
          "rgba8unorm"
        );
      },
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createSphereGeometry(32, 48));
  },
};

/** ===================== ELLIPSOID AXES (3 rings) ===================== **/

export const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  type: "EllipsoidAxes",

  defaults: {
    half_size: [0.5, 0.5, 0.5],
    quaternion: [0, 0, 0, 1],
  },

  getCount(elem) {
    // 3 rings per ellipsoid
    return (elem.centers.length / 3) * 3;
  },

  getFloatsPerInstance() {
    // position(3) + size(3) + quat(4) + color(3) + alpha(1) = 14
    return 14;
  },

  getFloatsPerPicking() {
    // same layout as Ellipsoid: 11
    return 11;
  },

  getCenters(elem) {
    // For sorting or bounding, etc. Usually the "per shape" centers,
    // not the 3x expanded. We'll just return the actual centers
    return elem.centers;
  },

  // offsets
  colorOffset: 10,
  alphaOffset: 13,

  fillRenderGeometry(elem, ringIndex, out, offset) {
    const i = Math.floor(ringIndex / 3);
    const constants = getElementConstants(this, elem);
    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3);
    } else {
      acopy(elem.half_sizes!, i * 3, out, offset + 3, 3);
    }

    // Quaternions
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4);
    }
  },

  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, ringIndex, out, offset, baseID) {
    const i = Math.floor(ringIndex / 3);
    const constants = getElementConstants(this, elem);
    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3);
    } else {
      acopy(elem.half_sizes!, i * 3, out, offset + 3, 3);
    }

    // Quaternions
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4);
    }

    // Use the ellipsoid index for picking, not the ring index
    out[offset + 10] = packID(baseID + i);
  },

  // We want ringIndex to use the same color index as the "i-th" ellipsoid
  getColorIndexForInstance(elem, ringIndex) {
    return Math.floor(ringIndex / 3);
  },

  renderConfig: {
    cullMode: "back",
    topology: "triangle-list",
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesShading",
      () => {
        return createTranslucentGeometryPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ringVertCode,
            fragmentShader: ringFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_INSTANCE_LAYOUT],
          },
          format,
          ellipsoidAxesSpec
        );
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesPicking",
      () => {
        return createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ringPickingVertCode,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_PICKING_INSTANCE_LAYOUT],
          },
          "rgba8unorm"
        );
      },
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createTorusGeometry(1.0, 0.03, 40, 12));
  },

  applyDecoration(out, instanceIndex, dec, floatsPerInstance) {
      // Apply to all three rings of the target ellipsoid
      for (let ring = 0; ring < 3; ring++) {
        const ringIndex = instanceIndex * 3 + ring;
        applyDefaultDecoration(out, ringIndex * floatsPerInstance, dec, this);
    }
  },
};

/** ===================== CUBOID ===================== **/

export interface CuboidComponentConfig extends BaseComponentConfig {
  type: "Cuboid";
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: number | [number, number, number];
  quaternions?: Float32Array;
  quaternion?: [number, number, number, number];
}

export const cuboidSpec: PrimitiveSpec<CuboidComponentConfig> = {
  type: "Cuboid",

  defaults: {
    half_size: [0.1, 0.1, 0.1],
    quaternion: [0, 0, 0, 1],
  },

  getCount(elem) {
    return elem.centers.length / 3;
  },

  getFloatsPerInstance() {
    // 3 pos + 3 size + 4 quat + 3 color + 1 alpha = 14
    return 14;
  },

  getFloatsPerPicking() {
    // 3 pos + 3 size + 4 quat + 1 pickID = 11
    return 11;
  },

  getCenters(elem) {
    return elem.centers;
  },

  colorOffset: 10,
  alphaOffset: 13,

  fillRenderGeometry(elem, i, out, offset) {
    const constants = getElementConstants(this, elem);

    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3);
    } else {
      acopy(elem.half_sizes as ArrayLike<number>, i * 3, out, offset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4);
    }

    // Color - use constant or per-instance value
    if (constants.color) {
      acopy(constants.color, 0, out, offset + 10, 3);
    } else {
      acopy(elem.colors!, i * 3, out, offset + 10, 3);
    }

    // Alpha - use constant or per-instance value
    out[offset + 13] = constants.alpha || elem.alphas![i];
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // multiply half_sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, i, out, offset, baseID) {
    const constants = getElementConstants(this, elem);

    // Position
    acopy(elem.centers, i * 3, out, offset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, offset + 3, 3);
    } else {
      acopy(elem.half_sizes as ArrayLike<number>, i * 3, out, offset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, offset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, offset + 6, 4);
    }

    // picking ID
    out[offset + 10] = packID(baseID + i);
  },

  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "CuboidShading",
      () => {
        return createTranslucentGeometryPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: cuboidVertCode,
            fragmentShader: cuboidFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [MESH_GEOMETRY_LAYOUT, CUBOID_INSTANCE_LAYOUT],
          },
          format,
          cuboidSpec
        );
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "CuboidPicking",
      () => {
        return createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: cuboidPickingVertCode,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [
              MESH_GEOMETRY_LAYOUT,
              CUBOID_PICKING_INSTANCE_LAYOUT,
            ],
            primitive: this.renderConfig,
          },
          "rgba8unorm"
        );
      },
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createCubeGeometry());
  },
};

/** ===================== LINE BEAMS ===================== **/

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: "LineBeams";
  points: Float32Array; // [x,y,z,lineIndex, x,y,z,lineIndex, ...]
  sizes?: Float32Array; // Per-line sizes
  size?: number; // Default size
}

/** We store a small WeakMap to "cache" the segment map for each config. */
const lineBeamsSegmentMap = new WeakMap<
  LineBeamsComponentConfig,
  {
    segmentMap: number[];
  }
>();

function prepareLineSegments(elem: LineBeamsComponentConfig): number[] {
  // If we already did it, return cached
  const cached = lineBeamsSegmentMap.get(elem);
  if (cached) return cached.segmentMap;

  const pointCount = elem.points.length / 4;
  const segmentIndices: number[] = [];

  for (let p = 0; p < pointCount - 1; p++) {
    const iCurr = elem.points[p * 4 + 3];
    const iNext = elem.points[(p + 1) * 4 + 3];
    if (iCurr === iNext) {
      segmentIndices.push(p);
    }
  }
  lineBeamsSegmentMap.set(elem, { segmentMap: segmentIndices });
  return segmentIndices;
}

function countSegments(elem: LineBeamsComponentConfig): number {
  return prepareLineSegments(elem).length;
}

export const lineBeamsSpec: PrimitiveSpec<LineBeamsComponentConfig> = {
  type: "LineBeams",

  defaults: {
    size: 0.02
  },

  getCount(elem) {
    return countSegments(elem);
  },

  getCenters(elem) {
    // Build array of each segment's midpoint, for sorting or bounding
    const segMap = prepareLineSegments(elem);
    const segCount = segMap.length;
    const centers = new Float32Array(segCount * 3);
    for (let s = 0; s < segCount; s++) {
      const p = segMap[s];
      const x0 = elem.points[p * 4 + 0];
      const y0 = elem.points[p * 4 + 1];
      const z0 = elem.points[p * 4 + 2];
      const x1 = elem.points[(p + 1) * 4 + 0];
      const y1 = elem.points[(p + 1) * 4 + 1];
      const z1 = elem.points[(p + 1) * 4 + 2];
      centers[s * 3 + 0] = (x0 + x1) * 0.5;
      centers[s * 3 + 1] = (y0 + y1) * 0.5;
      centers[s * 3 + 2] = (z0 + z1) * 0.5;
    }
    return centers;
  },

  getFloatsPerInstance() {
    // start(3) + end(3) + size(1) + color(3) + alpha(1) = 11
    return 11;
  },

  getFloatsPerPicking() {
    // start(3) + end(3) + size(1) + pickID(1) = 8
    return 8;
  },

  // offsets
  colorOffset: 7,
  alphaOffset: 10,

  /**
   * We want color/alpha to come from the line index (points[..+3]),
   * not from the segment index. So we define getColorIndexForInstance:
   */
  getColorIndexForInstance(elem, segmentIndex) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];
    // The line index is floor(points[p*4+3])
    return Math.floor(elem.points[p * 4 + 3]);
  },

  fillRenderGeometry(elem, segmentIndex, out, offset) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];
    const constants = getElementConstants(this, elem);

    // Start
    acopy(elem.points, p * 4, out, offset, 3);

    // End
    acopy(elem.points, (p + 1) * 4, out, offset + 3, 3);

    // Size
    const lineIndex = Math.floor(elem.points[p * 4 + 3]);
    out[offset + 6] = constants.size || elem.sizes![lineIndex];
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // only the size is at offset+6
    out[offset + 6] *= scaleFactor;
  },

  fillPickingGeometry(elem, segmentIndex, out, offset, baseID) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];
    const constants = getElementConstants(this, elem);

    // Start
    acopy(elem.points, p * 4, out, offset, 3);

    // End
    acopy(elem.points, (p + 1) * 4, out, offset + 3, 3);

    // Size
    const lineIndex = Math.floor(elem.points[p * 4 + 3]);
    out[offset + 6] = constants.size || elem.sizes![lineIndex];

    // pickID
    out[offset + 7] = packID(baseID + segmentIndex);
  },

  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "LineBeamsShading",
      () => {
        return createTranslucentGeometryPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: lineBeamVertCode,
            fragmentShader: lineBeamFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [MESH_GEOMETRY_LAYOUT, LINE_BEAM_INSTANCE_LAYOUT],
          },
          format,
          this
        );
      },
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "LineBeamsPicking",
      () => {
        return createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: lineBeamPickingVertCode,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_main",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [
              MESH_GEOMETRY_LAYOUT,
              LINE_BEAM_PICKING_INSTANCE_LAYOUT,
            ],
            primitive: this.renderConfig,
          },
          "rgba8unorm"
        );
      },
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createBeamGeometry());
  },
};

/** ===================== UNION TYPE FOR ALL COMPONENT CONFIGS ===================== **/

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;
