#ifndef PBR_METALLICWORKFLOW_GLSL
#define PBR_METALLICWORKFLOW_GLSL

#include "globals.glsl"
#include "random.glsl"
#include "common.glsl"

const float Pi = M_PI;
const float PiInv = 1.0 / Pi;

mat3 localRefMatrix(vec3 n) {
    vec3 t = (abs(n.y) > 0.9999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 b = normalize(cross(n, t));
    t = cross(b, n);
    return mat3(t, b, n);
}

vec3 localToWorld(vec3 n, vec3 v) {
    return normalize(localRefMatrix(n) * v);
}

vec3 sampleHemisphereCosine(vec3 n, vec2 r) {
    vec2 d = toConcentricDisk(r);
    float z = sqrt(1.0 - dot(d, d));
    return localToWorld(n, vec3(d, z));
}

float satDot(vec3 a, vec3 b) {
    return max(dot(a, b), 0.0);
}

float absDot(vec3 a, vec3 b) {
    return abs(dot(a, b));
}

vec3 FresnelSchlick(float cosTheta, vec3 f0) {
    float cos4 = 1.0 - cosTheta;
    cos4 *= cos4;
    cos4 *= cos4;
    return mix(f0, vec3(1.0), cos4 * (1.0 - cosTheta));
}

float SchlickG(float cosTheta, float alpha) {
    float a = alpha * 0.5;
    return cosTheta / (cosTheta * (1.0 - a) + a);
}

float SmithG(float cosWo, float cosWi, float alpha) {
    return SchlickG(abs(cosWo), alpha) * SchlickG(abs(cosWi), alpha);
}

float GTR2Distrib(float cosTheta, float alpha) {
    if (cosTheta < 1e-6) {
        return 0.0;
    }
    float aa = alpha * alpha;
    float nom = aa;
    float denom = cosTheta * cosTheta * (aa - 1.0) + 1.0;
    denom = denom * denom * Pi;
    return nom / denom;
}

float GTR2Pdf(vec3 n, vec3 m, vec3 wo, float alpha) {
    return GTR2Distrib(dot(n, m), alpha) * SchlickG(dot(n, wo), alpha) * absDot(m, wo) / absDot(n, wo);
}

vec3 GTR2Sample(vec3 n, vec3 wo, float alpha, vec2 r) {
    mat3 transMat = localRefMatrix(n);
    mat3 transInv = inverse(transMat);

    vec3 vh = normalize((transInv * wo) * vec3(alpha, alpha, 1.0));

    float lenSq = vh.x * vh.x + vh.y * vh.y;
    vec3 t = lenSq > 0.0 ? vec3(-vh.y, vh.x, 0.0) / sqrt(lenSq) : vec3(1.0, 0.0, 0.0);
    vec3 b = cross(vh, t);

    vec2 p = toConcentricDisk(r);
    float s = 0.5 * (vh.z + 1.0);
    p.y = (1.0 - s) * sqrt(1.0 - p.x * p.x) + s * p.y;

    vec3 h = t * p.x + b * p.y + vh * sqrt(max(0.0, 1.0 - dot(p, p)));
    h = vec3(h.x * alpha, h.y * alpha, max(0.0, h.z));
    return normalize(transMat * h);
}

vec3 metallicWorkflowBSDF(State state, vec3 n, vec3 wo, vec3 wi) {
    vec3 baseColor = state.mat.albedo;
    float roughness = state.mat.roughness;
    float metallic = state.mat.metallic;

    //float alpha = roughness * roughness;
    float alpha = roughness;
    vec3 h = normalize(wo + wi);

    float cosO = dot(n, wo);
    float cosI = dot(n, wi);
    if (cosI * cosO < 1e-7f) {
        return vec3(0.0);
    }

    vec3 f = FresnelSchlick(dot(h, wo), mix(vec3(.08f), baseColor, metallic));
    float g = SmithG(cosO, cosI, alpha);
    float d = GTR2Distrib(dot(n, h), alpha);

    return mix(baseColor * PiInv * (1.0 - metallic), vec3(g * d / (4.0 * cosI * cosO)), f);
}

float metallicWorkflowPdf(State state, vec3 n, vec3 wo, vec3 wi) {
    vec3 baseColor = state.mat.albedo;
    float roughness = state.mat.roughness;
    float metallic = state.mat.metallic;
    //float alpha = roughness * roughness;
    float alpha = roughness;

    vec3 h = normalize(wo + wi);
    return mix(
        satDot(n, wi) * PiInv,
        GTR2Pdf(n, h, wo, alpha) / (4.0 * absDot(h, wo)),
        1.0 / (2.0 - metallic)
    );
}

vec3 metallicWorkflowEval(State state, vec3 n, vec3 wo, vec3 wi, out float pdf) {
    vec3 baseColor = state.mat.albedo;
    float roughness = state.mat.roughness;
    float metallic = state.mat.metallic;

    //float alpha = roughness * roughness;
    float alpha = roughness;
    vec3 h = normalize(wo + wi);

    float cosO = dot(n, wo);
    float cosI = dot(n, wi);
    if (cosI * cosO < 1e-7f) {
        return vec3(0.0);
    }

    vec3 f = FresnelSchlick(dot(h, wo), mix(vec3(.08f), baseColor, metallic));
    float g = SmithG(cosO, cosI, alpha);
    float d = GTR2Distrib(dot(n, h), alpha);

    pdf = mix(satDot(n, wi) * PiInv, GTR2Pdf(n, h, wo, alpha) / (4.0 * absDot(h, wo)), 1.0 / (2.0 - metallic));
    return mix(baseColor * PiInv * (1.0 - metallic), vec3(g * d / (4.0 * cosI * cosO)), f);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 GgxSampling(float specularAlpha, float r1, float r2)
{
    float phi = r1 * 2.0 * M_PI;

    float cosTheta = sqrt((1.0 - r2) / (1.0 + (specularAlpha * specularAlpha - 1.0) * r2));
    float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float F_Schlick(float f0, float f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 EvalDielectricRefractionGltf(State state, vec3 V, vec3 N, vec3 L, vec3 H, inout float pdf)
{
    pdf = abs(dot(N, L));
    return state.mat.albedo;
}

float metallicWorkflowSample(State state, vec3 n, vec3 wo, out vec3 bsdf, out vec3 dir, inout RngStateType seed) {

    float transWeight = (1.0 - state.mat.metallic) * state.mat.transmission;

    float r1 = rand(seed);
    float r2 = rand(seed);

    // TODO
	// Trasmission
  //  if (rand(seed) < transWeight)
  //  {
  //      float eta = state.eta;

  //      float n1 = 1.0;
  //      float n2 = state.mat.ior;
  //      float R0 = (n1 - n2) / (n1 + n2);
  //      vec3  H = GgxSampling(state.mat.roughness, r1, r2);
  //      H = state.tangent * H.x + state.bitangent * H.y + n * H.z;
  //      float VdotH = dot(wo, H);
  //      float F = F_Schlick(R0 * R0, 1.0, VdotH);           // Reflection
  //      float discriminat = 1.0 - eta * eta * (1.0 - VdotH * VdotH);  // (Total internal reflection)

  //      if (state.mat.thinwalled)
  //      {
  //          // If inside surface, don't reflect
  //          if (dot(state.ffnormal, state.normal) < 0.0)
  //          {
  //              F = 0;
  //              discriminat = 0;
  //          }
  //          eta = 1.00;  // go through
  //      }

  //      // Reflection/Total internal reflection
  //      if (discriminat < 0.0 || rand(seed) < F)
  //      {
  //          dir = normalize(reflect(-wo, H));
  //          bsdf = vec3(10.0f, 0.0, 0.0);
  //      }
  //      else
  //      {
  //          // Find the pure refractive ray
  //          dir = normalize(refract(-wo, H, eta));

  //          // Cought rays perpendicular to surface, and simply continue
  //          if (isnan(dir.x) || isnan(dir.y) || isnan(dir.z))
  //          {
  //              dir = -wo;
  //          }
  //      }

  //      // Transmission
  //      float pdf;
  //      bsdf = EvalDielectricRefractionGltf(state, wo, n, dir, H, pdf);
		//return pdf;
  //  }

	// Metallic Workflow
    //else
    //{
    //    float roughness = state.mat.roughness;
    //    float metallic = state.mat.metallic;
    //    //float alpha = roughness * roughness;
    //    float alpha = roughness;

    //    if (rand(seed) > (1.0 / (2.0 - metallic))) {
    //        dir = sampleHemisphereCosine(n, vec2(rand(seed), rand(seed)));
    //    }
    //    else {
    //        vec3 h = GTR2Sample(n, wo, alpha, vec2(rand(seed), rand(seed)));
    //        dir = -reflect(wo, h);
    //    }

    //    if (dot(n, dir) < 0.0) {
    //        return InvalidPdf;
    //    }
    //    else {
    //        bsdf = metallicWorkflowBSDF(state, n, wo, dir);
    //        bsdf *= (1.0 - transWeight);
    //        return (1.0 - transWeight) * metallicWorkflowPdf(state, n, wo, dir);
    //    }
    //}

    float roughness = state.mat.roughness;
    float metallic = state.mat.metallic;
    //float alpha = roughness * roughness;
    float alpha = roughness;

    if (rand(seed) > (1.0 / (2.0 - metallic))) {
        dir = sampleHemisphereCosine(n, vec2(rand(seed), rand(seed)));
    }
    else {
        vec3 h = GTR2Sample(n, wo, alpha, vec2(rand(seed), rand(seed)));
        dir = -reflect(wo, h);
    }

    if (dot(n, dir) < 0.0) {
        return InvalidPdf;
    }
    else {
        bsdf = metallicWorkflowBSDF(state, n, wo, dir);
        bsdf *= (1.0 - transWeight);
        return (1.0 - transWeight) * metallicWorkflowPdf(state, n, wo, dir);
    }
}

vec3 metallicWorkflowSample(State state, vec3 n, vec3 wo, out vec3 dir, out float pdf, inout RngStateType seed) {
    vec3 bsdf;
    pdf = metallicWorkflowSample(state, n, wo, bsdf, dir, seed);
    return bsdf;
}

#endif