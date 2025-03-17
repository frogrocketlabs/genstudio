import genstudio.plot as Plot

shader = """

@group(0) @binding(0) var inputTex: texture_2d<f32>;
    @group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let dims = textureDimensions(inputTex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }
      let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
      // simple invert
      let outColor = vec4<f32>(1.0 - srcColor.r, 1.0 - srcColor.g, 1.0 - srcColor.b, 1.0);
      textureStore(outputTex, vec2<i32>(gid.xy), outColor);
    }

"""

(
    Plot.Import("path:notebooks/webgpu_compute.js", refer=["WebGPUVideoView"])
    | Plot.html(
        [
            Plot.js("WebGPUVideoView"),
            {
                "computeShader": shader,
                "showSourceVideo": True,
                "workgroupSize": [16, 16],
            },
        ]
    )
).save_html("notebooks/webgpu.html")
