// ----- "webgpuVideoCompute.js" -----
//
// A React component that processes webcam video through a WebGPU compute shader.
// The user provides the WGSL compute shader code as a string prop.
//
// Props:
// - computeShader: WGSL compute shader code as a string
// - width: Canvas width (default: 640)
// - height: Canvas height (default: 480)
// - showSourceVideo: Whether to show the source video (default: false)
// - workgroupSize: Array with two elements [x, y] for compute shader workgroup size (default: [8, 8])
//
// The compute shader should:
// - Use @group(0) @binding(0) for the input texture (texture_2d<f32>)
// - Use @group(0) @binding(1) for the output texture (texture_storage_2d<rgba8unorm, write>)
// - Have a main() function with @compute @workgroup_size(x, y) decorator matching workgroupSize prop
// - Take a @builtin(global_invocation_id) parameter to get pixel coordinates
// - Check texture bounds before processing pixels
//
// Example compute shader:
//
// @group(0) @binding(0) var inputTex : texture_2d<f32>;
// @group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;
//
// @compute @workgroup_size(8, 8)
// fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
//   let dims = textureDimensions(inputTex);
//   if (gid.x >= dims.x || gid.y >= dims.y) { return; }
//
//   let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
//   // Process srcColor here...
//   textureStore(outputTex, vec2<i32>(gid.xy), outColor);
// }
//

const { html } = genstudio.api;

export const WebGPUVideoView = ({computeShader, width = 640, height = 480, showSourceVideo = false, workgroupSize = [8, 8]}) => {
  const canvasId = React.useId();
  const videoProcessor = React.useRef(null);
  const frameRef = React.useRef(null);

  React.useEffect(() => {
    const init = async () => {
      videoProcessor.current = new WebGPUVideoCompute(canvasId, width, height, showSourceVideo);
      try {
        await videoProcessor.current.init();
        videoProcessor.current.setComputeShader(computeShader);
        videoProcessor.current.setWorkgroupSize(workgroupSize[0], workgroupSize[1]);

        const renderLoop = () => {
          frameRef.current = requestAnimationFrame(renderLoop);
          videoProcessor.current.renderFrame();
        };
        renderLoop();
      } catch (error) {
        console.error("WebGPU initialization failed:", error);
      }
    };

    init().catch(error => {
      console.error("Fatal error during initialization:", error);
    });

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      if (videoProcessor.current) {
        videoProcessor.current.cleanup();
      }
    };
  }, [computeShader, width, height, showSourceVideo, workgroupSize]);

  return html(['canvas', { id: canvasId, width, height }]);
};

class WebGPUVideoCompute {
  constructor(canvasId, videoWidth = 640, videoHeight = 480, showSourceVideo = false) {
    this.canvasId = canvasId;
    this.videoWidth = videoWidth;
    this.videoHeight = videoHeight;
    this.showSourceVideo = showSourceVideo;

    this.video = null;
    this.canvas = null;
    this.device = null;
    this.context = null;
    this.inputTexture = null;
    this.outputTexture = null;
    this.computePipeline = null;
    this.computeBindGroup = null;
    this.renderPipeline = null;
    this.renderBindGroup = null;
    this.sampler = null;

    this.workgroupSizeX = 8;
    this.workgroupSizeY = 8;

    // Offscreen canvas for capturing video frames.
    this.videoCanvas = document.createElement('canvas');
    this.videoCanvas.width = videoWidth;
    this.videoCanvas.height = videoHeight;
    this.videoCtx = this.videoCanvas.getContext('2d');

    if (showSourceVideo) {
      document.body.appendChild(this.videoCanvas);
      this.videoCanvas.style.position = 'fixed';
      this.videoCanvas.style.bottom = '10px';
      this.videoCanvas.style.right = '10px';
      this.videoCanvas.style.border = '1px solid red';
      this.videoCanvas.style.width = '160px';
      this.videoCanvas.style.height = '120px';
      this.videoCanvas.style.zIndex = '1000';
    }
  }

  async init() {
    await this._setupWebcam();
    await this._initWebGPU();
    this._createTextures();
    this._createRenderPipeline();
  }

  async _setupWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: this.videoWidth },
          height: { ideal: this.videoHeight }
        }
      });

      this.video = document.createElement('video');
      this.video.srcObject = stream;
      this.video.width = this.videoWidth;
      this.video.height = this.videoHeight;
      this.video.autoplay = true;
      this.video.playsInline = true;
      this.video.muted = true;

      return new Promise((resolve) => {
        this.video.onloadedmetadata = () => {
          this.video.play().then(() => {
            // Draw an initial frame.
            this.videoCtx.drawImage(this.video, 0, 0, this.videoWidth, this.videoHeight);
            resolve();
          }).catch(err => {
            console.error("Error playing video:", err);
            resolve();
          });
        };
      });
    } catch (error) {
      console.error("Webcam setup failed:", error);
      throw error;
    }
  }

  async _initWebGPU() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('Failed to get GPU adapter');
    }

    this.device = await adapter.requestDevice();

    this.canvas = document.getElementById(this.canvasId);
    if (!this.canvas) {
      throw new Error(`Canvas with ID ${this.canvasId} not found`);
    }

    this.context = this.canvas.getContext('webgpu');
    if (!this.context) {
      throw new Error('Failed to get WebGPU context');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: format,
      alphaMode: 'premultiplied',
    });
    this.renderFormat = format;
  }

  _createTextures() {
    const usage = GPUTextureUsage;
    const textureFormat = 'rgba8unorm';

    // Input texture: used by the compute shader (for textureLoad).
    this.inputTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: textureFormat,
      usage: usage.COPY_SRC | usage.COPY_DST | usage.TEXTURE_BINDING | usage.RENDER_ATTACHMENT,
      label: 'Input Texture'
    });

    // Output texture: written to by the compute shader.
    this.outputTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: textureFormat,
      usage: usage.STORAGE_BINDING | usage.TEXTURE_BINDING | usage.COPY_DST | usage.RENDER_ATTACHMENT,
      label: 'Output Texture'
    });
  }

  _createRenderPipeline() {
    const vertexShaderCode = /* wgsl */`
      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) texCoord: vec2<f32>,
      };

      @vertex
      fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        var output: VertexOutput;
        var positions = array<vec2<f32>, 6>(
          vec2<f32>(-1.0, -1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(1.0, 1.0)
        );
        var texCoords = array<vec2<f32>, 6>(
          vec2<f32>(0.0, 1.0),
          vec2<f32>(1.0, 1.0),
          vec2<f32>(0.0, 0.0),
          vec2<f32>(0.0, 0.0),
          vec2<f32>(1.0, 1.0),
          vec2<f32>(1.0, 0.0)
        );
        output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
        output.texCoord = texCoords[vertexIndex];
        return output;
      }
    `;

    const fragmentShaderCode = /* wgsl */`
      @group(0) @binding(0) var myTex: texture_2d<f32>;
      @group(0) @binding(1) var mySampler: sampler;

      @fragment
      fn fsMain(@location(0) texCoord: vec2<f32>) -> @location(0) vec4<f32> {
        return textureSample(myTex, mySampler, texCoord);
      }
    `;

    try {
      const renderModule = this.device.createShaderModule({
        code: vertexShaderCode + fragmentShaderCode
      });

      this.renderPipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: renderModule,
          entryPoint: 'vsMain',
        },
        fragment: {
          module: renderModule,
          entryPoint: 'fsMain',
          targets: [{ format: this.renderFormat }],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      this.sampler = this.device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });

      // Create the render bind group that will be reused
      const outputTextureView = this.outputTexture.createView();
      this.renderBindGroup = this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: outputTextureView },
          { binding: 1, resource: this.sampler },
        ],
      });

    } catch (error) {
      console.error("Error creating render pipeline:", error);
      throw error;
    }
  }

  setComputeShader(wgslCode) {
    try {
      if (!wgslCode || typeof wgslCode !== 'string') {
        throw new Error('Invalid shader code provided');
      }

      const module = this.device.createShaderModule({
        code: wgslCode,
        label: 'Compute Shader Module'
      });

      module.getCompilationInfo().then(info => {
        if (info.messages.length > 0) {
          console.warn("Shader compilation messages:", info.messages);
        }
      });

      this.computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module,
          entryPoint: 'main',
        },
        label: 'Compute Pipeline'
      });

      this.computeBindGroup = this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.inputTexture.createView() },
          { binding: 1, resource: this.outputTexture.createView() },
        ],
        label: 'Compute Bind Group'
      });
    } catch (error) {
      console.error("Error setting compute shader:", error);
    }
  }

  setWorkgroupSize(x, y) {
    this.workgroupSizeX = x;
    this.workgroupSizeY = y;
  }

  renderFrame = async () => {
    try {
      if (!this.video || this.video.readyState < 3 || this.video.paused) {
        console.log("Video not ready:",
                    this.video ? this.video.readyState : "no video",
                    "Paused:", this.video ? this.video.paused : "no video");
        return;
      }

      // Draw the current video frame onto the offscreen canvas.
      this.videoCtx.drawImage(this.video, 0, 0, this.videoWidth, this.videoHeight);

      try {
        // Create ImageBitmap from the canvas
        const imageBitmap = await createImageBitmap(this.videoCanvas);

        // Copy the ImageBitmap directly to the input texture
        this.device.queue.copyExternalImageToTexture(
          { source: imageBitmap },
          { texture: this.inputTexture },
          [this.videoWidth, this.videoHeight]
        );

      } catch (bitmapError) {
        console.error("Error handling bitmap:", bitmapError);
        // Fallback to the old method if bitmap fails
        const imageData = this.videoCtx.getImageData(0, 0, this.videoWidth, this.videoHeight);
        this.device.queue.writeTexture(
          { texture: this.inputTexture },
          imageData.data,
          { bytesPerRow: this.videoWidth * 4 },
          { width: this.videoWidth, height: this.videoHeight, depthOrArrayLayers: 1 }
        );
      }

      const commandEncoder = this.device.createCommandEncoder();

      // Run the compute shader pass.
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(this.computePipeline);
      computePass.setBindGroup(0, this.computeBindGroup);
      const wgX = Math.ceil(this.videoWidth / this.workgroupSizeX);
      const wgY = Math.ceil(this.videoHeight / this.workgroupSizeY);
      computePass.dispatchWorkgroups(wgX, wgY);
      computePass.end();

      const view = this.context.getCurrentTexture().createView();
      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      renderPass.setPipeline(this.renderPipeline);
      renderPass.setBindGroup(0, this.renderBindGroup);
      renderPass.draw(6, 1, 0, 0);
      renderPass.end();

      this.device.queue.submit([commandEncoder.finish()]);
    } catch (error) {
      console.error("Error in renderFrame:", error);
      console.error("Error stack:", error.stack);
    }
  }

  cleanup() {
    if (this.video && this.video.srcObject) {
      const tracks = this.video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    if (this.inputTexture) this.inputTexture.destroy();
    if (this.outputTexture) this.outputTexture.destroy();
  }
}
