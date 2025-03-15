import { quaternionShaderFunctions } from '../quaternion';
import {createVertexBufferLayout,
        cameraStruct,
        lightingConstants,
        lightingCalc,
        pickingVSOut
} from '../shaders';

export const RING_INSTANCE_LAYOUT = createVertexBufferLayout([
    [3, 'float32x3'], // instance center position
    [4, 'float32x3'], // instance size
    [5, 'float32x4'], // instance quaternion
    [6, 'float32x3'], // instance color
    [7, 'float32']    // instance alpha
  ], 'instance');

  export const RING_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout([
    [3, 'float32x3'], // position
    [4, 'float32x3'], // size
    [5, 'float32x4'], // quaternion
    [6, 'float32']    // pickID (now shared across rings)
  ], 'instance');

  export const RING_GEOMETRY_LAYOUT = createVertexBufferLayout([
    [0, 'float32x3'], // centerline position
    [1, 'float32x3'], // tube offset
    [2, 'float32x3']  // normal
  ], 'vertex');



  export const ringShaders = /*wgsl*/`
  ${cameraStruct}
  ${quaternionShaderFunctions}
  ${pickingVSOut}

  struct RenderVSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) worldPos: vec3<f32>,
    @location(3) normal: vec3<f32>
  };

  fn computeRingPosition(
    center: vec3<f32>,
    offset: vec3<f32>,
    position: vec3<f32>,
    size: vec3<f32>,
    quaternion: vec4<f32>
  ) -> vec3<f32> {
    // Apply non-uniform scaling to the centerline.
    let scaledCenter = quat_rotate(quaternion, center * size);

    // Compute a uniform scale for the tube offset (e.g. average of nonuniform scales).
    let uniformScale = (size.x + size.y + size.z) / 3.0;
    let scaledOffset = quat_rotate(quaternion, offset * uniformScale);

    // Final world position: instance position plus transformed center and offset.
    return position + scaledCenter + scaledOffset;
  }

  @vertex
  fn vs_render(
    @location(0) center: vec3<f32>,  // Centerline attribute (first 3 floats)
    @location(1) offset: vec3<f32>,  // Tube offset attribute (next 3 floats)
    @location(2) inNormal: vec3<f32>, // Precomputed normal (last 3 floats)
    @location(3) position: vec3<f32>,  // Instance center
    @location(4) size: vec3<f32>,      // Instance non-uniform scaling for ellipsoid
    @location(5) quaternion: vec4<f32>,// Instance rotation
    @location(6) inColor: vec3<f32>,   // Color attribute
    @location(7) alpha: f32            // Alpha attribute
  ) -> RenderVSOut {
    let worldPos = computeRingPosition(center, offset, position, size, quaternion);

    // For normals, we want the tube's offset direction unperturbed by nonuniform scaling.
    let worldNormal = quat_rotate(quaternion, normalize(offset));

    var out: RenderVSOut;
    out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
    out.color = inColor;
    out.alpha = alpha;
    out.worldPos = worldPos;
    out.normal = worldNormal;
    return out;
  }

  @vertex
  fn vs_pick(
    @location(0) center: vec3<f32>,  // Centerline attribute (first 3 floats)
    @location(1) offset: vec3<f32>,  // Tube offset attribute (next 3 floats)
    @location(2) inNormal: vec3<f32>, // Precomputed normal (last 3 floats)
    @location(3) position: vec3<f32>,  // Instance center
    @location(4) size: vec3<f32>,      // Instance non-uniform scaling for ellipsoid
    @location(5) quaternion: vec4<f32>,// Instance rotation
    @location(6) pickID: f32           // Picking ID
  ) -> VSOut {
    let worldPos = computeRingPosition(center, offset, position, size, quaternion);

    var out: VSOut;
    out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
    out.pickID = pickID;
    return out;
  }`;

  export const ellipsoidAxesFragCode = /*wgsl*/`
  ${cameraStruct}
  ${lightingConstants}
  ${lightingCalc}

  @fragment
  fn fs_main(
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) worldPos: vec3<f32>,
    @location(3) normal: vec3<f32>
  )-> @location(0) vec4<f32> {
    let litColor = calculateLighting(color, normal, worldPos);
    return vec4<f32>(litColor, alpha);
  }`;
