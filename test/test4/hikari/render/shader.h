#pragma once
namespace hikari {
  inline namespace render {
    struct Shader {

    };
    struct GraphicsShader                       : public Shader {};
    struct GraphicsVertexShader                 : public GraphicsShader {};
    struct GraphicsTessellationShader           : public GraphicsShader {};
    struct GraphicsGeometryShader               : public GraphicsShader {};
    struct GraphicsFramentShader                : public GraphicsShader {};
    struct GraphicsMeshShader                   : public GraphicsShader {};
    struct GraphicsTaskShader                   : public GraphicsShader {};
    struct GraphicsTessellationControlShader    : public GraphicsTessellationShader {};
    struct GraphicsTessellationEvaluationShader : public GraphicsTessellationShader {};
    struct ComputeShader                        : public Shader {};
    struct RayTracingShader                     : public Shader {};
    struct RayTracingRayGenerationShader        : public RayTracingShader {};
    struct RayTracingMissShader                 : public RayTracingShader {};
    struct RayTracingHitGroupShader             : public RayTracingShader {};
    struct RayTracingCallableShader             : public RayTracingShader {};
    struct RayTracingDirectCallableShader       : public RayTracingCallableShader {};
    struct RayTracingContinuationCallableShader : public RayTracingCallableShader {};
    struct RayTracingIntersectionShader         : public RayTracingHitGroupShader {};
    struct RayTracingClosestHitShader           : public RayTracingHitGroupShader {};
    struct RayTracingAnyHitShader               : public RayTracingHitGroupShader {};
  }
}
