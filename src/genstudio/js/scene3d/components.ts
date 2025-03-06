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

import { createCubeGeometry, createBeamGeometry, createSphereGeometry, createTorusGeometry } from './geometry';

import {packID} from './picking'

import {BaseComponentConfig, Decoration, PipelineCacheEntry, PrimitiveSpec, PipelineConfig, GeometryResource, GeometryData} from './types'

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

/** ===================== POINT CLOUD ===================== **/


export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: 'PointCloud';
  centers: Float32Array;
  sizes?: Float32Array;     // Per-point sizes
  size?: number;           // Default size, defaults to 0.02
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
  return createRenderPipeline(device, bindGroupLayout, {
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
  }, format);
}

const createBuffers = (device: GPUDevice, { vertexData, indexData }: GeometryData): GeometryResource => {
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



export const pointCloudSpec: PrimitiveSpec<PointCloudComponentConfig> = {
  type: 'PointCloud',
  getCount(elem) {
    return elem.centers.length / 3;
  },

  getFloatsPerInstance() {
    return 8;  // position(3) + size(1) + color(3) + alpha(1)
  },

  getFloatsPerPicking() {
    return 5;  // position(3) + size(1) + pickID(1)
  },

  getCenters(elem) {
    return elem.centers;
  },

  buildRenderData(elem, target, sortedIndices?: Uint32Array) {
    const count = elem.centers.length / 3;
    if(count === 0) return false;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const size = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= count ? elem.sizes : null;

    const {indices, indexToPosition} = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      // Position
      target[j*8+0] = elem.centers[i*3+0];
      target[j*8+1] = elem.centers[i*3+1];
      target[j*8+2] = elem.centers[i*3+2];

      // Size
      const pointSize = sizes ? sizes[i] : size;
      const scale = scales ? scales[i] : defaults.scale;
      target[j*8+3] = pointSize * scale;

      // Color
      if(colors) {
        target[j*8+4] = colors[i*3+0];
        target[j*8+5] = colors[i*3+1];
        target[j*8+6] = colors[i*3+2];
      } else {
        target[j*8+4] = defaults.color[0];
        target[j*8+5] = defaults.color[1];
        target[j*8+6] = defaults.color[2];
      }

      // Alpha
      target[j*8+7] = alphas ? alphas[i] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      if(dec.color) {
        target[j*8+4] = dec.color[0];
        target[j*8+5] = dec.color[1];
        target[j*8+6] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        target[j*8+7] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        target[j*8+3] *= dec.scale;  // Scale affects size
      }
    });

    return true;
  },

  buildPickingData(elem: PointCloudComponentConfig, target: Float32Array, baseID: number, sortedIndices?: Uint32Array): void {
    const count = elem.centers.length / 3;
    if(count === 0) return;

    const size = elem.size ?? 0.02;
    const hasValidSizes = elem.sizes && elem.sizes.length >= count;
    const sizes = hasValidSizes ? elem.sizes : null;
    const { scales } = getColumnarParams(elem, count);
    const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      // Position
      target[j*5+0] = elem.centers[i*3+0];
      target[j*5+1] = elem.centers[i*3+1];
      target[j*5+2] = elem.centers[i*3+2];
      // Size
      const pointSize = sizes?.[i] ?? size;
      const scale = scales ? scales[i] : 1.0;
      target[j*5+3] = pointSize * scale;
      // PickID - use baseID + local index
      target[j*5+4] = packID(baseID + i);
    }

    // Apply scale decorations
    applyDecorations(elem.decorations, count, (idx, dec) => {
      if(dec.scale !== undefined) {
        const j = indexToPosition ? indexToPosition[idx] : idx;
        if(j !== -1) {
          target[j*5+3] *= dec.scale;  // Scale affects size
        }
      }
    });
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
  half_sizes?: Float32Array;     // Per-ellipsoid half_sizes
  half_size?: [number, number, number]; // Default half_size, defaults to [1,1,1]
  quaternions?: Float32Array;   // Per-ellipsoid quaternions as quaternions [x,y,z,w]
  quaternion?: [number, number, number, number]; // Default quaternion quaternion [x,y,z,w]
}

export const ellipsoidSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  type: 'Ellipsoid',
  getCount(elem) {
    return elem.centers.length / 3;
  },

  getFloatsPerInstance() {
    return 14;  // position(3) + size(3) + quaternion(4) + color(3) + alpha(1)
  },

  getFloatsPerPicking() {
    return 11;  // position(3) + size(3) + quaternion(4) + pickID(1)
  },

  getCenters(elem) {return elem.centers;},

  buildRenderData(elem, target, sortedIndices?: Uint32Array) {
    const count = elem.centers.length / 3;
    if(count === 0) return false;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.half_size ?? [1, 1, 1];
    const half_sizes = elem.half_sizes && elem.half_sizes.length >= count * 3 ? elem.half_sizes : null;

    // Default identity quaternion [0,0,0,1]
    const defaultQuaternion = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    const {indices, indexToPosition} = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      const offset = j * 14; // Updated size: 3 (position) + 3 (size) + 4 (quaternion) + 3 (color) + 1 (alpha)

      // Position (location 2)
      target[offset+0] = elem.centers[i*3+0];
      target[offset+1] = elem.centers[i*3+1];
      target[offset+2] = elem.centers[i*3+2];

      // Size/half_sizes (location 3)
      const scale = scales ? scales[i] : defaults.scale;
      if(half_sizes) {
        target[offset+3] = half_sizes[i*3+0] * scale;
        target[offset+4] = half_sizes[i*3+1] * scale;
        target[offset+5] = half_sizes[i*3+2] * scale;
      } else {
        target[offset+3] = defaultRadius[0] * scale;
        target[offset+4] = defaultRadius[1] * scale;
        target[offset+5] = defaultRadius[2] * scale;
      }

      // Orientation (location 4)
      if(quaternions) {
        target[offset+6] = quaternions[i*4+0];
        target[offset+7] = quaternions[i*4+1];
        target[offset+8] = quaternions[i*4+2];
        target[offset+9] = quaternions[i*4+3];
      } else {
        target[offset+6] = defaultQuaternion[0];
        target[offset+7] = defaultQuaternion[1];
        target[offset+8] = defaultQuaternion[2];
        target[offset+9] = defaultQuaternion[3];
      }

      // Color (location 5)
      if(colors) {
        target[offset+10] = colors[i*3+0];
        target[offset+11] = colors[i*3+1];
        target[offset+12] = colors[i*3+2];
      } else {
        target[offset+10] = defaults.color[0];
        target[offset+11] = defaults.color[1];
        target[offset+12] = defaults.color[2];
      }

      // Alpha (location 6)
      target[offset+13] = alphas ? alphas[i] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      const offset = j * 14;
      if(dec.color) {
        target[offset+10] = dec.color[0];
        target[offset+11] = dec.color[1];
        target[offset+12] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        target[offset+13] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        target[offset+3] *= dec.scale;
        target[offset+4] *= dec.scale;
        target[offset+5] *= dec.scale;
      }
    });

    return true;
  },

  buildPickingData(elem: EllipsoidComponentConfig, target: Float32Array, baseID: number, sortedIndices?: Uint32Array): void {
    const count = elem.centers.length / 3;
    if(count === 0) return;

    const defaultSize = elem.half_size || [0.1, 0.1, 0.1];
    const sizes = elem.half_sizes && elem.half_sizes.length >= count * 3 ? elem.half_sizes : null;
    const { scales } = getColumnarParams(elem, count);

    // Default identity quaternion [0,0,0,1]
    const defaultQuaternion = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      const scale = scales ? scales[i] : 1;
      // Position
      target[j*11+0] = elem.centers[i*3+0];
      target[j*11+1] = elem.centers[i*3+1];
      target[j*11+2] = elem.centers[i*3+2];
      // Size
      target[j*11+3] = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
      target[j*11+4] = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
      target[j*11+5] = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;
      // Orientation
      if(quaternions) {
        target[j*11+6] = quaternions[i*4+0];
        target[j*11+7] = quaternions[i*4+1];
        target[j*11+8] = quaternions[i*4+2];
        target[j*11+9] = quaternions[i*4+3];
      } else {
        target[j*11+6] = defaultQuaternion[0];
        target[j*11+7] = defaultQuaternion[1];
        target[j*11+8] = defaultQuaternion[2];
        target[j*11+9] = defaultQuaternion[3];
      }
      // Picking ID
      target[j*11+10] = packID(baseID + i);
    }

    // Apply scale decorations
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      if (dec.scale !== undefined) {
        target[j*11+3] *= dec.scale;
        target[j*11+4] *= dec.scale;
        target[j*11+5] *= dec.scale;
      }
    });
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ellipsoidVertCode,
        fragmentShader: ellipsoidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
      }, format, ellipsoidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: ellipsoidPickingVertCode,
        fragmentShader: pickingFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createSphereGeometry(32, 48));
  }
};

/** ===================== ELLIPSOID AXES ===================== **/


export interface EllipsoidAxesComponentConfig extends BaseComponentConfig {
  type: 'EllipsoidAxes';
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: [number, number, number];  // Make optional since we have BaseComponentConfig defaults
  colors?: Float32Array;
  quaternions?: Float32Array;   // Per-ellipsoid quaternions as quaternions [x,y,z,w]
  quaternion?: [number, number, number, number]; // Default quaternion quaternion [x,y,z,w]
}

export const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidAxesComponentConfig> = {
  type: 'EllipsoidAxes',
  getCount(elem) {
    // Each ellipsoid has 3 rings
    return (elem.centers.length / 3) * 3;
  },

  getFloatsPerInstance() {
    return 14;  // position(3) + size(3) + quaternion(4) + color(3) + alpha(1)
  },

  getFloatsPerPicking() {
    return 11;  // position(3) + size(3) + quaternion(4) + pickID(1)
  },

  getCenters(elem) {return elem.centers},

  buildRenderData(elem, target, sortedIndices?: Uint32Array) {
    const count = elem.centers.length / 3;
    if(count === 0) return false;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.half_size ?? [1, 1, 1];
    const half_sizes = elem.half_sizes instanceof Float32Array && elem.half_sizes.length >= count * 3 ? elem.half_sizes : null;

    // Default identity quaternion [0,0,0,1]
    const defaultQuaternion = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    const {indices, indexToPosition} = getIndicesAndMapping(count, sortedIndices);

    const ringCount = count * 3;
    for(let j = 0; j < ringCount; j++) {
      const i = indices ? indices[Math.floor(j / 3)] : Math.floor(j / 3);
      const offset = j * 14; // Updated size: 3 (position) + 3 (size) + 4 (quaternion) + 3 (color) + 1 (alpha)

      // Position (location 2)
      target[offset+0] = elem.centers[i*3+0];
      target[offset+1] = elem.centers[i*3+1];
      target[offset+2] = elem.centers[i*3+2];

      // Size/half_sizes (location 3)
      const scale = scales ? scales[i] : defaults.scale;
      if(half_sizes) {
        target[offset+3] = half_sizes[i*3+0] * scale;
        target[offset+4] = half_sizes[i*3+1] * scale;
        target[offset+5] = half_sizes[i*3+2] * scale;
      } else {
        target[offset+3] = defaultRadius[0] * scale;
        target[offset+4] = defaultRadius[1] * scale;
        target[offset+5] = defaultRadius[2] * scale;
      }

      // Orientation (location 4)
      if(quaternions) {
        target[offset+6] = quaternions[i*4+0];
        target[offset+7] = quaternions[i*4+1];
        target[offset+8] = quaternions[i*4+2];
        target[offset+9] = quaternions[i*4+3];
      } else {
        target[offset+6] = defaultQuaternion[0];
        target[offset+7] = defaultQuaternion[1];
        target[offset+8] = defaultQuaternion[2];
        target[offset+9] = defaultQuaternion[3];
      }

      // Color (location 5)
      if(colors) {
        target[offset+10] = colors[i*3+0];
        target[offset+11] = colors[i*3+1];
        target[offset+12] = colors[i*3+2];
      } else {
        target[offset+10] = defaults.color[0];
        target[offset+11] = defaults.color[1];
        target[offset+12] = defaults.color[2];
      }

      // Alpha (location 6)
      target[offset+13] = alphas ? alphas[i] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      // For each decorated ellipsoid, update all 3 of its rings
      for(let ring = 0; ring < 3; ring++) {
        const arrIdx = j*3 + ring;
        if(dec.color) {
          target[arrIdx*14+10] = dec.color[0];
          target[arrIdx*14+11] = dec.color[1];
          target[arrIdx*14+12] = dec.color[2];
        }
        if(dec.alpha !== undefined) {
          target[arrIdx*14+13] = dec.alpha;
        }
        if(dec.scale !== undefined) {
          target[arrIdx*14+3] *= dec.scale;
          target[arrIdx*14+4] *= dec.scale;
          target[arrIdx*14+5] *= dec.scale;
        }
      }
    });

    return true;
  },

  buildPickingData(elem: EllipsoidAxesComponentConfig, target: Float32Array, baseID: number, sortedIndices?: Uint32Array): void {
    const count = elem.centers.length / 3;
    if(count === 0) return;

    const defaultRadius = elem.half_size ?? [1, 1, 1];

    // Default identity quaternion [0,0,0,1]
    const defaultQuaternion = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      const rx = elem.half_sizes?.[i*3+0] ?? defaultRadius[0];
      const ry = elem.half_sizes?.[i*3+1] ?? defaultRadius[1];
      const rz = elem.half_sizes?.[i*3+2] ?? defaultRadius[2];

      // Get quaternion for this ellipsoid
      let qx, qy, qz, qw;
      if (quaternions) {
        qx = quaternions[i*4+0];
        qy = quaternions[i*4+1];
        qz = quaternions[i*4+2];
        qw = quaternions[i*4+3];
      } else {
        qx = defaultQuaternion[0];
        qy = defaultQuaternion[1];
        qz = defaultQuaternion[2];
        qw = defaultQuaternion[3];
      }

      const thisID = packID(baseID + i);

      for(let ring = 0; ring < 3; ring++) {
        const idx = j*3 + ring;
        // Position
        target[idx*11+0] = cx;
        target[idx*11+1] = cy;
        target[idx*11+2] = cz;
        // Size
        target[idx*11+3] = rx;
        target[idx*11+4] = ry;
        target[idx*11+5] = rz;
        // Orientation
        target[idx*11+6] = qx;
        target[idx*11+7] = qy;
        target[idx*11+8] = qz;
        target[idx*11+9] = qw;
        // Picking ID
        target[idx*11+10] = thisID;
      }
    }

    applyDecorations(elem.decorations, count, (idx, dec) => {
      if (!dec.scale) return;
      const j = indexToPosition ? indexToPosition[idx] : idx;
      // For each decorated ellipsoid, update all 3 of its rings
      for(let ring = 0; ring < 3; ring++) {
        const arrIdx = j*3 + ring;
        target[arrIdx*11+3] *= dec.scale;
        target[arrIdx*11+4] *= dec.scale;
        target[arrIdx*11+5] *= dec.scale;
      }
    });
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ringVertCode,
        fragmentShader: ringFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_INSTANCE_LAYOUT],
        blend: {} // Use defaults
      }, format, ellipsoidAxesSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: ringPickingVertCode,
        fragmentShader: pickingFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, RING_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createTorusGeometry(1.0, 0.03, 40, 12));
  }
};

/** ===================== CUBOID ===================== **/


export interface CuboidComponentConfig extends BaseComponentConfig {
  type: 'Cuboid';
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: [number, number, number];
  quaternions?: Float32Array;   // Per-cuboid quaternions as quaternions [x,y,z,w]
  quaternion?: [number, number, number, number]; // Default quaternion quaternion [x,y,z,w]
}

export const cuboidSpec: PrimitiveSpec<CuboidComponentConfig> = {
  type: 'Cuboid',
  getCount(elem){
    return elem.centers.length / 3;
  },
  getFloatsPerInstance() {
    return 14;  // position(3) + size(3) + quaternion(4) + color(3) + alpha(1)
  },
  getFloatsPerPicking() {
    return 11;  // position(3) + size(3) + quaternion(4) + pickID(1)
  },
  getCenters(elem) { return elem.centers},

  buildRenderData(elem, target, sortedIndices?: Uint32Array) {
    const count = elem.centers.length / 3;
    if(count === 0) return false;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);
    const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);

    const defaultHalfSize = elem.half_size || [0.1, 0.1, 0.1];
    const half_sizes = elem.half_sizes && elem.half_sizes.length >= count * 3 ? elem.half_sizes : null;

    // Default identity quaternion [0,0,0,1]
    const defaultOrientation = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      const scale = scales ? scales[i] : defaults.scale;

      // Get half_sizes with scale
      const sx = (half_sizes ? half_sizes[i*3+0] : defaultHalfSize[0]) * scale;
      const sy = (half_sizes ? half_sizes[i*3+1] : defaultHalfSize[1]) * scale;
      const sz = (half_sizes ? half_sizes[i*3+2] : defaultHalfSize[2]) * scale;

      // Get quaternion
      let qx, qy, qz, qw;
      if (quaternions) {
        qx = quaternions[i*4+0];
        qy = quaternions[i*4+1];
        qz = quaternions[i*4+2];
        qw = quaternions[i*4+3];
      } else {
        qx = defaultOrientation[0];
        qy = defaultOrientation[1];
        qz = defaultOrientation[2];
        qw = defaultOrientation[3];
      }

      // Get colors
      let cr: number, cg: number, cb: number;
      if (colors) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      } else {
        cr = defaults.color[0];
        cg = defaults.color[1];
        cb = defaults.color[2];
      }
      const alpha = alphas ? alphas[i] : defaults.alpha;

      // Fill array
      const idx = j * 14; // Updated size: 3 (position) + 3 (size) + 4 (quaternion) + 3 (color) + 1 (alpha)
      target[idx+0] = cx;
      target[idx+1] = cy;
      target[idx+2] = cz;
      target[idx+3] = sx;
      target[idx+4] = sy;
      target[idx+5] = sz;
      target[idx+6] = qx;
      target[idx+7] = qy;
      target[idx+8] = qz;
      target[idx+9] = qw;
      target[idx+10] = cr;
      target[idx+11] = cg;
      target[idx+12] = cb;
      target[idx+13] = alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;  // Get the position where this index ended up
      if(dec.color) {
        target[j*14+10] = dec.color[0];
        target[j*14+11] = dec.color[1];
        target[j*14+12] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        target[j*14+13] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        target[j*14+3] *= dec.scale;
        target[j*14+4] *= dec.scale;
        target[j*14+5] *= dec.scale;
      }
    });

    return true;
  },
  buildPickingData(elem: CuboidComponentConfig, target: Float32Array, baseID: number, sortedIndices?: Uint32Array): void {
    const count = elem.centers.length / 3;
    if(count === 0) return;

    const defaultHalfSize = elem.half_size || [0.1, 0.1, 0.1];
    const half_sizes = elem.half_sizes && elem.half_sizes.length >= count * 3 ? elem.half_sizes : null;
    const { scales } = getColumnarParams(elem, count);

    // Default identity quaternion [0,0,0,1]
    const defaultOrientation = elem.quaternion ?? [0, 0, 0, 1];
    const quaternions = elem.quaternions && elem.quaternions.length >= count * 4 ? elem.quaternions : null;

    const { indices, indexToPosition } = getIndicesAndMapping(count, sortedIndices);

    for(let j = 0; j < count; j++) {
      const i = indices ? indices[j] : j;
      const scale = scales ? scales[i] : 1;
      // Position
      target[j*11+0] = elem.centers[i*3+0];
      target[j*11+1] = elem.centers[i*3+1];
      target[j*11+2] = elem.centers[i*3+2];
      // Size
      target[j*11+3] = (half_sizes ? half_sizes[i*3+0] : defaultHalfSize[0]) * scale;
      target[j*11+4] = (half_sizes ? half_sizes[i*3+1] : defaultHalfSize[1]) * scale;
      target[j*11+5] = (half_sizes ? half_sizes[i*3+2] : defaultHalfSize[2]) * scale;
      // Orientation
      if(quaternions) {
        target[j*11+6] = quaternions[i*4+0];
        target[j*11+7] = quaternions[i*4+1];
        target[j*11+8] = quaternions[i*4+2];
        target[j*11+9] = quaternions[i*4+3];
      } else {
        target[j*11+6] = defaultOrientation[0];
        target[j*11+7] = defaultOrientation[1];
        target[j*11+8] = defaultOrientation[2];
        target[j*11+9] = defaultOrientation[3];
      }
      // Picking ID
      target[j*11+10] = packID(baseID + i);
    }

    // Apply scale decorations
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      if (dec.scale !== undefined) {
        target[j*11+3] *= dec.scale;
        target[j*11+4] *= dec.scale;
        target[j*11+5] *= dec.scale;
      }
    });
  },
  renderConfig: {
    cullMode: 'none',  // Cuboids need to be visible from both sides
    topology: 'triangle-list'
  },
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "CuboidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: cuboidVertCode,
        fragmentShader: cuboidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, CUBOID_INSTANCE_LAYOUT]
      }, format, cuboidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "CuboidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: cuboidPickingVertCode,
        fragmentShader: pickingFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, CUBOID_PICKING_INSTANCE_LAYOUT],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createCubeGeometry());
  }
};

/******************************************************
 *  LineBeams Type
 ******************************************************/

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: 'LineBeams';
  points: Float32Array;  // [x,y,z,i, x,y,z,i, ...]
  sizes?: Float32Array;     // Per-line sizes
  size?: number;         // Default size, defaults to 0.02
}

function countSegments(points: Float32Array): number {
  const pointCount = points.length / 4;
  if (pointCount < 2) return 0;

  let segCount = 0;
  for (let p = 0; p < pointCount - 1; p++) {
    const iCurr = points[p * 4 + 3];
    const iNext = points[(p+1) * 4 + 3];
    if (iCurr === iNext) {
      segCount++;
    }
  }
  return segCount;
}

export const lineBeamsSpec: PrimitiveSpec<LineBeamsComponentConfig> = {
  type: 'LineBeams',
  getCount(elem) {
    return countSegments(elem.points);
  },

  getCenters(elem) {
    const segCount = this.getCount(elem);
        const centers = new Float32Array(segCount * 3);
        let segIndex = 0;
        const pointCount = elem.points.length / 4;

        for(let p = 0; p < pointCount - 1; p++) {
          const iCurr = elem.points[p * 4 + 3];
          const iNext = elem.points[(p+1) * 4 + 3];
          if(iCurr !== iNext) continue;

          centers[segIndex*3+0] = (elem.points[p*4+0] + elem.points[(p+1)*4+0]) * 0.5;
          centers[segIndex*3+1] = (elem.points[p*4+1] + elem.points[(p+1)*4+1]) * 0.5;
          centers[segIndex*3+2] = (elem.points[p*4+2] + elem.points[(p+1)*4+2]) * 0.5;
          segIndex++;
        }
        return centers
  },

  getFloatsPerInstance() {
    return 11;
  },

  getFloatsPerPicking() {
    return 8;  // startPos(3) + endPos(3) + size(1) + pickID(1)
  },

  buildRenderData(elem, target, sortedIndices?: Uint32Array) {
    const segCount = this.getCount(elem);
    if(segCount === 0) return false;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, segCount);

    // First pass: build segment mapping
    const segmentMap = new Array(segCount);
    let segIndex = 0;

    const pointCount = elem.points.length / 4;
      for(let p = 0; p < pointCount - 1; p++) {
        const iCurr = elem.points[p * 4 + 3];
        const iNext = elem.points[(p+1) * 4 + 3];
        if(iCurr !== iNext) continue;

        // Store mapping from segment index to point index
      segmentMap[segIndex] = p;
        segIndex++;
    }

    const defaultSize = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= segCount ? elem.sizes : null;

    const {indices, indexToPosition} = getIndicesAndMapping(segCount, sortedIndices);

    for(let j = 0; j < segCount; j++) {
      const i = indices ? indices[j] : j;
      const p = segmentMap[i];
      const lineIndex = Math.floor(elem.points[p * 4 + 3]);

      // Start point
      target[j*11+0] = elem.points[p * 4 + 0];
      target[j*11+1] = elem.points[p * 4 + 1];
      target[j*11+2] = elem.points[p * 4 + 2];

      // End point
      target[j*11+3] = elem.points[(p+1) * 4 + 0];
      target[j*11+4] = elem.points[(p+1) * 4 + 1];
      target[j*11+5] = elem.points[(p+1) * 4 + 2];

      // Size with scale
      const scale = scales ? scales[lineIndex] : defaults.scale;
      target[j*11+6] = (sizes ? sizes[lineIndex] : defaultSize) * scale;

      // Colors
      if(colors) {
        target[j*11+7] = colors[lineIndex*3+0];
        target[j*11+8] = colors[lineIndex*3+1];
        target[j*11+9] = colors[lineIndex*3+2];
      } else {
        target[j*11+7] = defaults.color[0];
        target[j*11+8] = defaults.color[1];
        target[j*11+9] = defaults.color[2];
      }

      target[j*11+10] = alphas ? alphas[lineIndex] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, segCount, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;
      if(dec.color) {
        target[j*11+7] = dec.color[0];
        target[j*11+8] = dec.color[1];
        target[j*11+9] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        target[j*11+10] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        target[j*11+6] *= dec.scale;
      }
    });

    return true;
  },

  buildPickingData(elem: LineBeamsComponentConfig, target: Float32Array, baseID: number, sortedIndices?: Uint32Array): void {
    const segCount = this.getCount(elem);
    if(segCount === 0) return;

    const defaultSize = elem.size ?? 0.02;

    // First pass: build segment mapping
    const segmentMap = new Array(segCount);
    let segIndex = 0;

    const pointCount = elem.points.length / 4;
    for(let p = 0; p < pointCount - 1; p++) {
      const iCurr = elem.points[p * 4 + 3];
      const iNext = elem.points[(p+1) * 4 + 3];
      if(iCurr !== iNext) continue;

      // Store mapping from segment index to point index
      segmentMap[segIndex] = p;
      segIndex++;
    }

    const { indices, indexToPosition } = getIndicesAndMapping(segCount, sortedIndices);

    for(let j = 0; j < segCount; j++) {
      const i = indices ? indices[j] : j;
      const p = segmentMap[i];
      const lineIndex = Math.floor(elem.points[p * 4 + 3]);
      let size = elem.sizes?.[lineIndex] ?? defaultSize;
      const scale = elem.scales?.[lineIndex] ?? 1.0;

      size *= scale;

      // Apply decorations that affect size
      applyDecorations(elem.decorations, lineIndex + 1, (idx, dec) => {
        idx = indexToPosition ? indexToPosition[idx] : idx;
        if(idx === lineIndex && dec.scale !== undefined) {
          size *= dec.scale;
        }
      });

      const base = j * 8;
      target[base + 0] = elem.points[p * 4 + 0];     // start.x
      target[base + 1] = elem.points[p * 4 + 1];     // start.y
      target[base + 2] = elem.points[p * 4 + 2];     // start.z
      target[base + 3] = elem.points[(p+1) * 4 + 0]; // end.x
      target[base + 4] = elem.points[(p+1) * 4 + 1]; // end.y
      target[base + 5] = elem.points[(p+1) * 4 + 2]; // end.z
      target[base + 6] = size;                        // size
      target[base + 7] = packID(baseID + i);
    }
  },

  // Standard triangle-list, cull as you like
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "LineBeamsShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: lineBeamVertCode,   // defined below
        fragmentShader: lineBeamFragCode, // defined below
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_INSTANCE_LAYOUT ],
      }, format, this),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "LineBeamsPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: lineBeamPickingVertCode,
        fragmentShader: pickingFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_PICKING_INSTANCE_LAYOUT ],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createBeamGeometry());
  }
};

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | EllipsoidAxesComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;
