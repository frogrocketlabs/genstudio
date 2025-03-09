export interface PipelineCacheEntry {
    pipeline: GPURenderPipeline;
    device: GPUDevice;
  }

export interface PrimitiveSpec<ConfigType extends BaseComponentConfig> {
    /**
     * The type/name of this primitive spec
     */
    type: string;

    /**
     * Default values for the primitive's properties
     */
    defaults?: ElementConstants;

    /**
     * Returns the number of instances in this component.
     */
    getCount(elem: ConfigType): number;

    /**
     * Returns the number of floats needed per instance for render data.
     */
    getFloatsPerInstance(): number;

    /**
     * Returns the number of floats needed per instance for picking data.
     */
    getFloatsPerPicking(): number;

    /**
     * Returns the centers of all instances in this component.
     * Used for transparency sorting and distance calculations.
     * @returns Object containing centers array and stride, or undefined if not applicable
     */
    getCenters(elem: ConfigType): Float32Array;

    /**
     * Offset for color data in the vertex buffer
     */
    colorOffset: number;

    /**
     * Offset for alpha data in the vertex buffer
     */
    alphaOffset: number;

    /**
     * Fills geometry data for rendering a single instance.
     * @param component The component containing instance data
     * @param instanceIndex Index of the instance to fill data for
     * @param out Output Float32Array to write data to
     * @param offset Offset in the output array to start writing
     * @param scale Scale factor to apply to the instance
     */
    fillRenderGeometry(elem: ConfigType, i: number, out: Float32Array, offset: number): void;

    /**
     * Applies a scale decoration to an instance.
     * @param out Output Float32Array containing instance data
     * @param offset Offset in the output array where instance data starts
     * @param scaleFactor Scale factor to apply
     */
    applyDecorationScale(out: Float32Array, offset: number, scaleFactor: number): void;

    /**
     * Fills geometry data for picking a single instance.
     * @param component The component containing instance data
     * @param instanceIndex Index of the instance to fill data for
     * @param out Output Float32Array to write data to
     * @param offset Offset in the output array to start writing
     * @param baseID Base ID for picking
     * @param scale Scale factor to apply to the instance
     */
    fillPickingGeometry(elem: ConfigType, i: number, out: Float32Array, offset: number, baseID: number): void;

    /**
     * Optional method to get the color index for an instance.
     * Used when the color index is different from the instance index.
     * @param component The component containing instance data
     * @param instanceIndex Index of the instance to get color for
     * @returns The index to use for color lookup
     */
    getColorIndexForInstance?(elem: ConfigType, i: number): number;

    /**
     * Optional method to apply decorations to an instance.
     * Used when decoration needs special handling beyond default color/alpha/scale.
     * @param out Output Float32Array containing instance data
     * @param instanceIndex Index of the instance being decorated
     * @param dec The decoration to apply
     * @param floatsPerInstance Number of floats per instance in the buffer
     */
    applyDecoration?(out: Float32Array, instanceIndex: number, dec: Decoration, floatsPerInstance: number): void;

    /**
     * Default WebGPU rendering configuration for this primitive type.
     * Specifies face culling and primitive topology.
     */
    renderConfig: {
      cullMode: GPUCullMode;
      topology: GPUPrimitiveTopology;
    };

    /**
     * Creates or retrieves a cached WebGPU render pipeline for this primitive.
     * @param device The WebGPU device
     * @param bindGroupLayout Layout for uniform bindings
     * @param cache Pipeline cache to prevent duplicate creation
     */
    getRenderPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>
    ): GPURenderPipeline;

    /**
     * Creates or retrieves a cached WebGPU pipeline for picking.
     * @param device The WebGPU device
     * @param bindGroupLayout Layout for uniform bindings
     * @param cache Pipeline cache to prevent duplicate creation
     */
    getPickingPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>
    ): GPURenderPipeline;

    /**
     * Creates the base geometry buffers needed for this primitive type.
     * These buffers are shared across all instances of the primitive.
     */
    createGeometryResource(device: GPUDevice): GeometryResource;
  }

export interface Decoration {
    indexes: number[];
    color?: [number, number, number];
    alpha?: number;
    scale?: number;
  }


export interface ElementConstants {
  half_size?: number[] | Float32Array | number;
  quaternion?: number[] | Float32Array;
  size?: number;
  color?: [number, number, number] | Float32Array;
  alpha?: number;
  scale?: number;
}

export interface BaseComponentConfig {
    constants?: ElementConstants;
    /**
     * Per-instance RGB color values as a Float32Array of RGB triplets.
     * Each instance requires 3 consecutive values in the range [0,1].
     */
    colors?: Float32Array;

    /**
     * Per-instance alpha (opacity) values.
     * Each value should be in the range [0,1].
     */
    alphas?: Float32Array;

    /**
     * Per-instance scale multipliers.
     * These multiply the base size/radius of each instance.
     */
    scales?: Float32Array;

    /**
     * Default RGB color applied to all instances without specific colors.
     * Values should be in range [0,1]. Defaults to [1,1,1] (white).
     */
    color?: [number, number, number];

    /**
     * Default alpha (opacity) for all instances without specific alpha.
     * Should be in range [0,1]. Defaults to 1.0.
     */
    alpha?: number;

    /**
     * Callback fired when the mouse hovers over an instance.
     * The index parameter is the instance index, or null when hover ends.
     */
    onHover?: (index: number|null) => void;

    /**
     * Callback fired when an instance is clicked.
     * The index parameter is the clicked instance index.
     */
    onClick?: (index: number) => void;

    /**
     * Optional array of decorations to apply to specific instances.
     * Decorations can override colors, alpha, and scale for individual instances.
     */
    decorations?: Decoration[];
  }

  export interface VertexBufferLayout {
    arrayStride: number;
    stepMode?: GPUVertexStepMode;
    attributes: {
      shaderLocation: number;
      offset: number;
      format: GPUVertexFormat;
    }[];
  }

  export interface PipelineConfig {
    vertexShader: string;
    fragmentShader: string;
    vertexEntryPoint: string;
    fragmentEntryPoint: string;
    bufferLayouts: VertexBufferLayout[];
    primitive?: {
      topology?: GPUPrimitiveTopology;
      cullMode?: GPUCullMode;
    };
    blend?: {
      color?: GPUBlendComponent;
      alpha?: GPUBlendComponent;
    };
    depthStencil?: {
      format: GPUTextureFormat;
      depthWriteEnabled: boolean;
      depthCompare: GPUCompareFunction;
    };
    colorWriteMask?: number;  // Use number instead of GPUColorWrite
  }

export interface GeometryData {
  vertexData: Float32Array;
  indexData: Uint16Array | Uint32Array;
}

export interface GeometryResource {
  vb: GPUBuffer;
  ib: GPUBuffer;
  indexCount: number;
  vertexCount: number;
}

export interface GeometryResources {
  PointCloud: GeometryResource | null;
  Ellipsoid: GeometryResource | null;
  Cuboid: GeometryResource | null;
  LineBeams: GeometryResource | null;
}

export interface BufferInfo {
    buffer: GPUBuffer;
    offset: number;
    stride: number;
  }

  export interface RenderObject {
    pipeline?: GPURenderPipeline;
    vertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
    indexBuffer?: GPUBuffer;
    vertexCount?: number;
    indexCount?: number;
    instanceCount?: number;

    pickingPipeline?: GPURenderPipeline;
    pickingVertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
    pickingIndexBuffer?: GPUBuffer;
    pickingVertexCount?: number;
    pickingIndexCount?: number;
    pickingInstanceCount?: number;

    componentIndex: number;
    pickingDataStale: boolean;

    // Arrays owned by this RenderObject, reallocated only when count changes
    cachedRenderData: Float32Array;   // Make non-optional since all components must have render data
    cachedPickingData: Float32Array;  // Make non-optional since all components must have picking data
    lastRenderCount: number;          // Make non-optional since we always need to track this

    // Temporary sorting state
    sortedIndices?: Uint32Array;
    distances?: Float32Array;

    // Cache for partitioned indices to reduce GC pressure
    cachedPartitions?: Uint32Array[];

    componentOffsets: ComponentOffset[];

    /** Reference to the primitive spec that created this render object */
    spec: PrimitiveSpec<any>;
  }

  export interface RenderObjectCache {
    [key: string]: RenderObject;  // Key is componentType, value is the most recent render object
  }

  export interface DynamicBuffers {
    renderBuffer: GPUBuffer;
    pickingBuffer: GPUBuffer;
    renderOffset: number;  // Current offset into render buffer
    pickingOffset: number; // Current offset into picking buffer
  }

  export interface ComponentOffset {
    componentIdx: number; // The index of the component in your overall component list.
    start: number;        // The first instance index in the combined buffer for this component.
    count: number;        // How many instances this component contributed.
  }

export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: 'Ellipsoid' | 'EllipsoidAxes';
  centers: Float32Array | number[];
  half_sizes?: Float32Array | number[];
  half_size?: [number, number, number] | number;
  quaternions?: Float32Array | number[];
  quaternion?: [number, number, number, number];
  colors?: Float32Array | number[];
  color?: [number, number, number];
  alphas?: Float32Array | number[];
  alpha?: number;
  fill_mode?: 'Solid' | 'MajorWireframe' | 'DenseWireframe';
}

// Remove EllipsoidAxesComponentConfig since it's merged into EllipsoidComponentConfig

/** ===================== UNION TYPE FOR ALL COMPONENT CONFIGS ===================== **/

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;
