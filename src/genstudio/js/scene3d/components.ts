import {
  LIGHTING,
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
  RING_PICKING_INSTANCE_LAYOUT
} from './shaders';

import {
  createCubeGeometry,
  createBeamGeometry,
  createSphereGeometry,
  createTorusGeometry
} from './geometry';

import { packID } from './picking';

import {
  BaseComponentConfig,
  Decoration,
  PipelineCacheEntry,
  PrimitiveSpec,
  PipelineConfig,
  GeometryResource,
  GeometryData
} from './types';

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

  function getBaseDefaults(config: Partial<BaseComponentConfig>): Required<Omit<BaseComponentConfig, 'colors' | 'alphas' | 'scales' | 'decorations' | 'onHover' | 'onClick'>> {
  return {
    color: config.color ?? [1, 1, 1],
    alpha: config.alpha ?? 1.0,
    scale: config.scale ?? 1.0,
  };
}

  function getColumnarParams(elem: BaseComponentConfig, count: number): {colors: Float32Array|null, alphas: Float32Array|null, scales: Float32Array|null} {
    const hasValidColors = elem.colors instanceof Float32Array && elem.colors.length >= count * 3;
  const hasValidAlphas = elem.alphas instanceof Float32Array && elem.alphas.length >= count;
  const hasValidScales = elem.scales instanceof Float32Array && elem.scales.length >= count;

  return {
      colors: hasValidColors ? (elem.colors as Float32Array) : null,
      alphas: hasValidAlphas ? (elem.alphas as Float32Array) : null,
      scales: hasValidScales ? (elem.scales as Float32Array) : null
  };
}

/** Helper function to handle sorted indices and position mapping */
function getIndicesAndMapping(count: number, sortedIndices?: Uint32Array): {
  indices: Uint32Array | null,  // Change to Uint32Array
  indexToPosition: Uint32Array | null
} {
  if (!sortedIndices) {
    return {
      indices: null,
      indexToPosition: null
    };
  }

  // Only create mapping if we have sorted indices
  const indexToPosition = new Uint32Array(count);
  for(let j = 0; j < count; j++) {
    indexToPosition[sortedIndices[j]] = j;
  }

  return {
    indices: sortedIndices,
    indexToPosition
  };
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

  // Retrieve base defaults (color, alpha, scale=1.0 fallback)
  const defaults = getBaseDefaults(elem);

  // Columnar arrays for colors/alphas/scales
  const { colors, alphas, scales } = getColumnarParams(elem, count);

  const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);
  const floatsPerInstance = spec.getFloatsPerInstance();

  for (let j = 0; j < count; j++) {
    const i = indices ? indices[j] : j;
    const offset = j * floatsPerInstance;

    // Combine shape's scale with user's per-instance scale array or default
    const finalScale = scales ? scales[i] : defaults.scale;

    // Let the shape fill the geometry portion (positions, sizes, quaternions, etc.)
    spec.fillRenderGeometry(elem, i, out, offset, finalScale);

    // Color / alpha usage is handled here
    const colorIndex = spec.getColorIndexForInstance
      ? spec.getColorIndexForInstance(elem, i)
      : i;
    const r = colors ? colors[colorIndex * 3 + 0] : defaults.color[0];
    const g = colors ? colors[colorIndex * 3 + 1] : defaults.color[1];
    const b = colors ? colors[colorIndex * 3 + 2] : defaults.color[2];
    const a = alphas ? alphas[colorIndex] : defaults.alpha;

    out[offset + spec.colorOffset] = r;
    out[offset + spec.colorOffset + 1] = g;
    out[offset + spec.colorOffset + 2] = b;
    out[offset + spec.alphaOffset] = a;
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

  const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);
  const floatsPerPicking = spec.getFloatsPerPicking();

  // Do the main fill
  for (let j = 0; j < count; j++) {
    const i = indices ? indices[j] : j;
    const offset = j * floatsPerPicking;
    // Let the shape fill the picking geometry (positions, orientation, pickID)
    spec.fillPickingGeometry(elem, i, out, offset, baseID, 1.0); // scale=1.0 initially
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
  cache: Map<string, PipelineCacheEntry>  // This will be the instance cache
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
    bindGroupLayouts: [bindGroupLayout]
  });

  // Get primitive configuration with defaults
  const primitiveConfig = {
    topology: config.primitive?.topology || 'triangle-list',
    cullMode: config.primitive?.cullMode || 'back'
  };

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: config.vertexShader }),
      entryPoint: config.vertexEntryPoint,
      buffers: config.bufferLayouts
    },
    fragment: {
      module: device.createShaderModule({ code: config.fragmentShader }),
      entryPoint: config.fragmentEntryPoint,
      targets: [{
          format,
          writeMask: config.colorWriteMask ?? GPUColorWrite.ALL,
          ...(config.blend && {
            blend: {
              color: config.blend.color || {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha'
              },
              alpha: config.blend.alpha || {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha'
              }
            }
          })
      }]
    },
    primitive: primitiveConfig,
    depthStencil: config.depthStencil || {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  });
}

function createTranslucentGeometryPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any>  // Take the primitive spec instead of just type
): GPURenderPipeline {
  return createRenderPipeline(
    device,
    bindGroupLayout,
    {
      ...config,
      primitive: primitiveSpec.renderConfig,
      blend: {
        color: {
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add'
        },
        alpha: {
          srcFactor: 'one',
          dstFactor: 'one-minus-src-alpha',
          operation: 'add'
        }
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less'
      }
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
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(vb, 0, vertexData);

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(ib, 0, indexData);

  // Each vertex has 6 floats (position + normal)
  const vertexCount = vertexData.length / 6;

  return {
    vb,
    ib,
    indexCount: indexData.length,
    vertexCount
  };
};

/** ===================== POINT CLOUD ===================== **/

export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: 'PointCloud';
  centers: Float32Array;
  sizes?: Float32Array; // Per-point sizes
  size?: number; // Default size, defaults to 0.02
}

export const pointCloudSpec: PrimitiveSpec<PointCloudComponentConfig> = {
  type: 'PointCloud',

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
  fillRenderGeometry(elem, i, out, offset, scale) {
    // Position
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    const defaultSize = elem.size ?? 0.02;
    const sizesValid = elem.sizes instanceof Float32Array && elem.sizes.length > i;
    const pointSize = sizesValid ? elem.sizes![i] : defaultSize;
    out[offset + 3] = pointSize * scale; // size
  },

  // For decorations that scale the point size
  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
  },

  // fillPickingGeometry
  fillPickingGeometry(elem, i, out, offset, baseID, scale) {
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    const defaultSize = elem.size ?? 0.02;
    const sizesValid = elem.sizes instanceof Float32Array && elem.sizes.length > i;
    const pointSize = sizesValid ? elem.sizes![i] : defaultSize;
    out[offset + 3] = pointSize * scale;

    // pickID
    out[offset + 4] = packID(baseID + i);
  },
  // Rendering configuration
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  // Pipeline creation methods
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
        device,
      "PointCloudShading",
      () => createRenderPipeline(device, bindGroupLayout, {
          vertexShader: billboardVertCode,
          fragmentShader: billboardFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_main',
          bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_INSTANCE_LAYOUT],
          primitive: this.renderConfig,
          blend: {
            color: {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            }
          },
          depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less'
          }
      }, format),
      cache
      );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
        device,
      "PointCloudPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
          vertexShader: billboardPickingVertCode,
          fragmentShader: pickingFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_pick',
          bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
      );
  },

  createGeometryResource(device) {
    return createBuffers(device, {
      vertexData: new Float32Array([
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
        -0.5,  0.5, 0.0,     0.0, 0.0, 1.0,
         0.5,  0.5, 0.0,     0.0, 0.0, 1.0
      ]),
      indexData: new Uint16Array([0,1,2, 2,1,3])
    });
  }
};

/** ===================== ELLIPSOID ===================== **/


export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: 'Ellipsoid';
  centers: Float32Array;
  half_sizes?: Float32Array; // Per-ellipsoid half_sizes
  half_size?: [number, number, number]; // Default half_size
  quaternions?: Float32Array; // Per-ellipsoid quaternions [x,y,z,w]
  quaternion?: [number, number, number, number]; // Default quaternion
}

export const ellipsoidSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  type: 'Ellipsoid',

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

  getCenters(elem) {return elem.centers;},

  // Where color/alpha go
  colorOffset: 10,
  alphaOffset: 13,

  fillRenderGeometry(elem, i, out, offset, scale) {
    // Position
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    // half_sizes
    const defaultRadius = elem.half_size ?? [1, 1, 1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    const rx = half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultRadius[0];
    const ry = half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultRadius[1];
    const rz = half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultRadius[2];

    out[offset + 3] = rx * scale;
    out[offset + 4] = ry * scale;
    out[offset + 5] = rz * scale;

    // Orientation
    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // Multiply the half_sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, i, out, offset, baseID, scale) {
    // position
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    // half_sizes (now using scale parameter)
    const defaultRadius = elem.half_size ?? [0.1, 0.1, 0.1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    out[offset + 3] = (half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultRadius[0]) * scale;
    out[offset + 4] = (half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultRadius[1]) * scale;
    out[offset + 5] = (half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultRadius[2]) * scale;

    // Orientation
    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];

    // picking ID
    out[offset + 10] = packID(baseID + i);
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(device, 'EllipsoidShading', () => {
      return createTranslucentGeometryPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: ellipsoidVertCode,
          fragmentShader: ellipsoidFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_main',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
        },
        format,
        ellipsoidSpec
      );
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(device, 'EllipsoidPicking', () => {
      return createRenderPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: ellipsoidPickingVertCode,
          fragmentShader: pickingFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_pick',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_PICKING_INSTANCE_LAYOUT]
        },
        'rgba8unorm'
      );
    }, cache);
  },

  createGeometryResource(device) {
    return createBuffers(device, createSphereGeometry(32, 48));
  }
};

/** ===================== ELLIPSOID AXES (3 rings) ===================== **/

export interface EllipsoidAxesComponentConfig extends BaseComponentConfig {
  type: 'EllipsoidAxes';
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: [number, number, number];
  colors?: Float32Array;
  quaternions?: Float32Array;
  quaternion?: [number, number, number, number];
}

export const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidAxesComponentConfig> = {
  type: 'EllipsoidAxes',

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

  fillRenderGeometry(elem, ringIndex, out, offset, scale) {
    // Get the base ellipsoid index by dividing by 3
    const i = Math.floor(ringIndex / 3);

    // Position, size, and quaternion stay the same
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    const defaultSize = elem.half_size ?? [1, 1, 1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    const rx = half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultSize[0];
    const ry = half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultSize[1];
    const rz = half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultSize[2];

    out[offset + 3] = rx * scale;
    out[offset + 4] = ry * scale;
    out[offset + 5] = rz * scale;

    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];
  },

  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, ringIndex, out, offset, baseID, scale) {
    // ringIndex = (ellipsoidIndex*3 + ringNumber)
    const i = Math.floor(ringIndex / 3);  // Get the ellipsoid index

    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    const defaultRadius = elem.half_size ?? [1, 1, 1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    const rx = (half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultRadius[0]) * scale;
    const ry = (half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultRadius[1]) * scale;
    const rz = (half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultRadius[2]) * scale;

    out[offset + 3] = rx;
    out[offset + 4] = ry;
    out[offset + 5] = rz;

    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];

    // Use the ellipsoid index for picking, not the ring index
    out[offset + 10] = packID(baseID + i);
  },

  // We want ringIndex to use the same color index as the "i-th" ellipsoid
  getColorIndexForInstance(elem, ringIndex) {
    return Math.floor(ringIndex / 3);
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(device, 'EllipsoidAxesShading', () => {
      return createTranslucentGeometryPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: ringVertCode,
          fragmentShader: ringFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_main',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_INSTANCE_LAYOUT]
        },
        format,
        ellipsoidAxesSpec
      );
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(device, 'EllipsoidAxesPicking', () => {
      return createRenderPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: ringPickingVertCode,
          fragmentShader: pickingFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_pick',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_PICKING_INSTANCE_LAYOUT]
        },
        'rgba8unorm'
      );
    }, cache);
  },

  createGeometryResource(device) {
    return createBuffers(device, createTorusGeometry(1.0, 0.03, 40, 12));
  },

  applyDecoration(out, instanceIndex, dec, floatsPerInstance) {
    // Apply to all three rings of the target ellipsoid
    for (let ring = 0; ring < 3; ring++) {
      const ringIndex = (instanceIndex * 3) + ring;
      applyDefaultDecoration(out, ringIndex * floatsPerInstance, dec, this);
    }
  },
};

/** ===================== CUBOID ===================== **/

export interface CuboidComponentConfig extends BaseComponentConfig {
  type: 'Cuboid';
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: [number, number, number];
  quaternions?: Float32Array; // [x,y,z,w]
  quaternion?: [number, number, number, number];
}

export const cuboidSpec: PrimitiveSpec<CuboidComponentConfig> = {
  type: 'Cuboid',

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

  fillRenderGeometry(elem, i, out, offset, scale) {
    // position
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    // half_sizes
    const defaultSize = elem.half_size || [0.1, 0.1, 0.1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    const sx = (half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultSize[0]) * scale;
    const sy = (half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultSize[1]) * scale;
    const sz = (half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultSize[2]) * scale;

    out[offset + 3] = sx;
    out[offset + 4] = sy;
    out[offset + 5] = sz;

    // orientation
    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // multiply half_sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(elem, i, out, offset, baseID, scale) {
    // position
    out[offset + 0] = elem.centers[i * 3 + 0];
    out[offset + 1] = elem.centers[i * 3 + 1];
    out[offset + 2] = elem.centers[i * 3 + 2];

    // size (now using scale parameter)
    const defaultSize = elem.half_size || [0.1, 0.1, 0.1];
    const half_sizesOK = elem.half_sizes && elem.half_sizes.length >= (i + 1) * 3;
    out[offset + 3] = (half_sizesOK ? elem.half_sizes![i * 3 + 0] : defaultSize[0]) * scale;
    out[offset + 4] = (half_sizesOK ? elem.half_sizes![i * 3 + 1] : defaultSize[1]) * scale;
    out[offset + 5] = (half_sizesOK ? elem.half_sizes![i * 3 + 2] : defaultSize[2]) * scale;

    // orientation
    const defaultQuat = elem.quaternion ?? [0, 0, 0, 1];
    const quatsOK = elem.quaternions && elem.quaternions.length >= (i + 1) * 4;
    out[offset + 6] = quatsOK ? elem.quaternions![i * 4 + 0] : defaultQuat[0];
    out[offset + 7] = quatsOK ? elem.quaternions![i * 4 + 1] : defaultQuat[1];
    out[offset + 8] = quatsOK ? elem.quaternions![i * 4 + 2] : defaultQuat[2];
    out[offset + 9] = quatsOK ? elem.quaternions![i * 4 + 3] : defaultQuat[3];

    // pickID
    out[offset + 10] = packID(baseID + i);
  },

  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(device, 'CuboidShading', () => {
      return createTranslucentGeometryPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: cuboidVertCode,
          fragmentShader: cuboidFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_main',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, CUBOID_INSTANCE_LAYOUT]
        },
        format,
        cuboidSpec
      );
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(device, 'CuboidPicking', () => {
      return createRenderPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: cuboidPickingVertCode,
          fragmentShader: pickingFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_pick',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, CUBOID_PICKING_INSTANCE_LAYOUT],
          primitive: this.renderConfig
        },
        'rgba8unorm'
      );
    }, cache);
  },

  createGeometryResource(device) {
    return createBuffers(device, createCubeGeometry());
  }
};

/** ===================== LINE BEAMS ===================== **/

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: 'LineBeams';
  points: Float32Array; // [x,y,z,lineIndex, x,y,z,lineIndex, ...]
  sizes?: Float32Array; // Per-line sizes
  size?: number; // Default size
}

/** We store a small WeakMap to "cache" the segment map for each config. */
const lineBeamsSegmentMap = new WeakMap<LineBeamsComponentConfig, {
  segmentMap: number[];
}>();

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
  type: 'LineBeams',

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

  fillRenderGeometry(elem, segmentIndex, out, offset, scale) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];

    // Start
    out[offset + 0] = elem.points[p * 4 + 0];
    out[offset + 1] = elem.points[p * 4 + 1];
    out[offset + 2] = elem.points[p * 4 + 2];

    // End
    out[offset + 3] = elem.points[(p + 1) * 4 + 0];
    out[offset + 4] = elem.points[(p + 1) * 4 + 1];
    out[offset + 5] = elem.points[(p + 1) * 4 + 2];

    // Size
    const lineIndex = Math.floor(elem.points[p * 4 + 3]);
    const defaultSize = elem.size ?? 0.02;
    const userSize =
      elem.sizes && elem.sizes.length > lineIndex
        ? elem.sizes[lineIndex]
        : defaultSize;

    out[offset + 6] = userSize * scale;
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // only the size is at offset+6
    out[offset + 6] *= scaleFactor;
  },

  fillPickingGeometry(elem, segmentIndex, out, offset, baseID, scale) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];

    // Start
    out[offset + 0] = elem.points[p * 4 + 0];
    out[offset + 1] = elem.points[p * 4 + 1];
    out[offset + 2] = elem.points[p * 4 + 2];

    // End
    out[offset + 3] = elem.points[(p + 1) * 4 + 0];
    out[offset + 4] = elem.points[(p + 1) * 4 + 1];
    out[offset + 5] = elem.points[(p + 1) * 4 + 2];

    // Size (now using scale parameter)
    const lineIndex = Math.floor(elem.points[p * 4 + 3]);
    const defaultSize = elem.size ?? 0.02;
    const userSize =
      elem.sizes && elem.sizes.length > lineIndex
        ? elem.sizes[lineIndex]
        : defaultSize;
    out[offset + 6] = userSize * scale;

    // pickID
    out[offset + 7] = packID(baseID + segmentIndex);
  },

  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(device, 'LineBeamsShading', () => {
      return createTranslucentGeometryPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: lineBeamVertCode,
          fragmentShader: lineBeamFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_main',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, LINE_BEAM_INSTANCE_LAYOUT]
        },
        format,
        this
      );
    }, cache);
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(device, 'LineBeamsPicking', () => {
      return createRenderPipeline(
        device,
        bindGroupLayout,
        {
          vertexShader: lineBeamPickingVertCode,
          fragmentShader: pickingFragCode,
          vertexEntryPoint: 'vs_main',
          fragmentEntryPoint: 'fs_pick',
          bufferLayouts: [MESH_GEOMETRY_LAYOUT, LINE_BEAM_PICKING_INSTANCE_LAYOUT],
          primitive: this.renderConfig
        },
        'rgba8unorm'
      );
    }, cache);
  },

  createGeometryResource(device) {
    return createBuffers(device, createBeamGeometry());
  }
};

/** ===================== UNION TYPE FOR ALL COMPONENT CONFIGS ===================== **/

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | EllipsoidAxesComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;
