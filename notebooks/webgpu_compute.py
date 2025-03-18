import genstudio.plot as Plot
from genstudio.plot import js

invert = {
    "shader": """
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> tint: vec4<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(inputTex);
  if (gid.x >= dims.x || gid.y >= dims.y) {
    return;
  }
  let srcColor = textureLoad(inputTex, vec2<i32>(gid.xy), 0);
  // simple invert followed by tinting
  let invertedColor = vec4<f32>(1.0 - srcColor.r, 1.0 - srcColor.g, 1.0 - srcColor.b, 1.0);
  let tintedColor = invertedColor * tint;
  textureStore(outputTex, vec2<i32>(gid.xy), tintedColor);
}
"""
}

pixelate = {
    "shader": """
// Texture bindings for input/output and uniform buffer
// uniforms.w contains the pixel block size
@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: vec4<f32>;

// Each workgroup handles one pixel block
@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) group_id : vec3<u32>
) {
    let dims = textureDimensions(inputTex);

    // Extract tint and block size from uniforms
    let tint = vec4<f32>(uniforms.x, uniforms.y, uniforms.z, 1.0);
    let blockSize = max(2.0, uniforms.w);

    // Calculate block coordinates
    let blockX = group_id.x * u32(blockSize);
    let blockY = group_id.y * u32(blockSize);

    // Skip if this block is completely outside texture bounds
    if (blockX >= dims.x || blockY >= dims.y) {
        return;
    }

    // Calculate average color for this block
    var avgColor = vec4<f32>(0.0);
    var count = 0u;

    // Sum up all pixels in this block
    let blockEndX = min(blockX + u32(blockSize), dims.x);
    let blockEndY = min(blockY + u32(blockSize), dims.y);

    for (var y = blockY; y < blockEndY; y++) {
        for (var x = blockX; x < blockEndX; x++) {
            avgColor += textureLoad(inputTex, vec2<i32>(i32(x), i32(y)), 0);
            count += 1u;
        }
    }

    // Apply average color to all pixels in this block
    if (count > 0u) {
        avgColor = (avgColor / f32(count)) * tint;

        // Write the averaged color to all pixels in this block
        for (var y = blockY; y < blockEndY; y++) {
            for (var x = blockX; x < blockEndX; x++) {
                textureStore(outputTex, vec2<i32>(i32(x), i32(y)), avgColor);
            }
        }
    }
}
""",
    # Create one workgroup per block
    "customDispatch": js(
        "[Math.ceil($state.width / $state.pixelBlockSize), Math.ceil($state.height / $state.pixelBlockSize)]"
    ),
    "workgroupSize": [1, 2],
}

(
    Plot.Import("path:notebooks/webgpu_compute.js", refer_all=True)
    | Plot.Slider("intendedPixelBlockSize", init=8, range=[2, 200])
    | Plot.initialState(
        {
            "width": 640,
            "height": 480,
            "tint": [1.0, 1.0, 1.0],
            "pixelBlockSize": js(
                """
                             const w = $state.width;
                             const numBlocks = Math.ceil(w/$state.intendedPixelBlockSize);
                             return w/numBlocks;
                             """,
                expression=False,
            ),
            "currentShader": "pixelate",
            "getCurrentShader": js("() => $state.shaders[$state.currentShader]"),
            "shaders": {"pixelate": pixelate, "invert": invert},
        }
    )
    | [
        [
            "div",
            {"className": "flex space-x-4 mb-4"},
            [
                [
                    "button",
                    {
                        "onClick": js("(e) => { $state.currentShader = 'pixelate'; }"),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js("$state.currentShader === 'pixelate'"),
                    },
                    "Pixelate",
                ],
                [
                    "button",
                    {
                        "onClick": js("(e) => { $state.currentShader = 'invert'; }"),
                        "className": "px-4 py-2 rounded border hover:bg-gray-200 data-[selected=true]:bg-blue-500 data-[selected=true]:text-white",
                        "data-selected": js("$state.currentShader === 'invert'"),
                    },
                    "Invert",
                ],
            ],
        ],
        [
            js("WebGPUVideoView"),
            {
                "computeShader": js("$state.getCurrentShader().shader"),
                "showSourceVideo": True,
                "workgroupSize": js(
                    "$state.getCurrentShader().workgroupSize || [16, 16]"
                ),
                "dispatchScale": js("$state.getCurrentShader().dispatchScale"),
                "customDispatch": js("$state.getCurrentShader().customDispatch"),
                "uniforms": js(
                    "[$state.tint[0], $state.tint[1], $state.tint[2], $state.pixelBlockSize]"
                ),
                "width": js("$state.width"),
                "height": js("$state.height"),
            },
        ],
        [
            js("colorScrubber"),
            {
                "value": js("$state.tint || [0,0,0]"),
                "onInput": js("(e) => { $state.tint = e.target.value; }"),
            },
        ],
    ]
).save_html("notebooks/webgpu.html")
