#define ENVMAP 1
#define RR 0        // Using russian roulette
#define RR_DEPTH 0  // Minimum depth

#include "pbr_metallicworkflow.glsl"
#include "gltf_material.glsl"
#include "env_sampling.glsl"
#include "shade_state.glsl"
#include "punctual.glsl"

float dummyPdf;

bool IsPdfInvalid(float p) {
    return p <= 1e-8 || isnan(p);
}

bool Occlusion(Ray ray, State state, float dist) {
    return AnyHit(ray, dist - abs(ray.origin.x - state.position.x) -
        abs(ray.origin.y - state.position.y) -
        abs(ray.origin.z - state.position.z));
}

vec3 BSDF(State state, vec3 V, vec3 N, vec3 L) {
    return metallicWorkflowBSDF(state, N, V, L);
}

float Pdf(State state, vec3 V, vec3 N, vec3 L) {
    return metallicWorkflowPdf(state, N, V, L);
}

vec3 Eval(State state, vec3 V, vec3 N, vec3 L, inout float pdf) {
    return metallicWorkflowEval(state, N, V, L, pdf);
}

vec3 Sample(State state, vec3 V, vec3 N, inout vec3 L, inout float pdf, inout RngStateType seed) {
    return metallicWorkflowSample(state, N, V, vec3(rand(seed), rand(seed), rand(seed)), L, pdf);
}

vec3 EnvRadiance(vec3 dir) {
    if (_sunAndSky.in_use == 1)
        return sun_and_sky(_sunAndSky, dir) * rtxState.hdrMultiplier;
    else {
        vec2 uv = GetSphericalUv(dir);
        return texture(environmentTexture, uv).rgb * rtxState.hdrMultiplier;
    }
}

float EnvPdf(vec3 dir) {
    float pdf;
    if (_sunAndSky.in_use == 1) {
        pdf = 0.5;
    }
    else {
        vec2 uv = GetSphericalUv(dir);
        pdf = luminance(texture(environmentTexture, uv).rgb) * rtxState.envMapLuminIntegInv;
    }
    return pdf * rtxState.environmentProb;
}

vec3 EnvEval(vec3 dir, out float pdf) {
    if (_sunAndSky.in_use == 1) {
        pdf = 0.5 * rtxState.environmentProb;
        return sun_and_sky(_sunAndSky, dir) * rtxState.hdrMultiplier;
    }
    else {
        vec2 uv = GetSphericalUv(dir);
        vec3 radiance = texture(environmentTexture, uv).rgb;
        pdf = luminance(radiance) * rtxState.envMapLuminIntegInv * rtxState.environmentProb;
        return radiance;
    }
}

vec3 LightEval(State state, float dist, vec3 dir, out float pdf) {
    float lightProb = (1.0 - rtxState.environmentProb);

    GltfShadeMaterial mat = materials[state.matID];
    vec3 emission = mat.emissiveFactor;

    pdf = luminance(emission) * rtxState.lightLuminIntegInv * lightProb;
    pdf *= dist * dist / absDot(state.ffnormal, dir);

    if (mat.emissiveTexture > -1) {
        vec2 uv = state.texCoord;
        emission *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(mat.emissiveTexture)], uv, 0)).rgb;
    }
    return emission/* / area*/;
}

vec2 SampleTriangleUniform(vec3 v0, vec3 v1, vec3 v2) {
    float ru = rand(prd.seed);
    float rv = rand(prd.seed);
    float r = sqrt(rv);
    float u = 1.0 - r;
    float v = ru * r;
    return vec2(u, v);
}

float TriangleArea(vec3 v0, vec3 v1, vec3 v2) {
    return length(cross(v1 - v0, v2 - v0)) * 0.5;
}

float SampleTriangleLight(vec3 x, out LightSample lightSample) {
    if (lightBufInfo.trigLightSize == 0) {
        return InvalidPdf;
    }

    int id = min(int(float(lightBufInfo.trigLightSize) * rand(prd.seed)), int(lightBufInfo.trigLightSize) - 1);

    if (rand(prd.seed) > trigLights[id].impSamp.q) {
        id = trigLights[id].impSamp.alias;
    }

    TrigLight light = trigLights[id];

    vec3 v0 = light.v0;
    vec3 v1 = light.v1;
    vec3 v2 = light.v2;

    vec3 normal = cross(v1 - v0, v2 - v0);
    float area = length(normal) * 0.5;
    normal = normalize(normal);

    vec2 baryCoord = SampleTriangleUniform(v0, v1, v2);
    vec3 y = baryCoord.x * v0 + baryCoord.y * v1 + (1 - baryCoord.x - baryCoord.y) * v2;

    GltfShadeMaterial mat = materials[light.matIndex];
    vec3 emission = mat.emissiveFactor;
    if (mat.emissiveTexture > -1) {
        vec2 uv = baryCoord.x * light.uv0 + baryCoord.y * light.uv1 + (1 - baryCoord.x - baryCoord.y) * light.uv2;
        emission *= SRGBtoLINEAR(textureLod(texturesMap[nonuniformEXT(mat.emissiveTexture)], uv, 0)).rgb;
    }
    vec3 dir = y - x;
    float dist = length(dir);
    lightSample.Li = emission/* / area*/;
    lightSample.wi = dir / dist;
    lightSample.dist = dist;
    return light.impSamp.pdf * (dist * dist) / (area * abs(dot(lightSample.wi, normal)));
}

float SamplePuncLight(vec3 x, out LightSample lightSample) {
    if (lightBufInfo.puncLightSize == 0) {
        return InvalidPdf;
    }

    int id = min(int(float(lightBufInfo.puncLightSize) * rand(prd.seed)), int(lightBufInfo.puncLightSize) - 1);

    if (rand(prd.seed) > puncLights[id].impSamp.q) {
        id = puncLights[id].impSamp.alias;
    }

    PuncLight light = puncLights[id];
    vec3 dir = light.position - x;
    float dist = length(dir);
    lightSample.Li = light.color * light.intensity / (dist * dist);
    lightSample.wi = dir / dist;
    lightSample.dist = dist;
    return light.impSamp.pdf;
}

float SampleDirectLightNoVisibility(vec3 pos, out LightSample lightSample) {
    float rnd = rand(prd.seed);
    if (rnd < rtxState.environmentProb) {
        // Sample environment
        vec4 dirAndPdf = EnvSample(lightSample.Li);
        if (IsPdfInvalid(dirAndPdf.w)) {
            return InvalidPdf;
        }
        lightSample.wi = dirAndPdf.xyz;
        lightSample.dist = INFINITY;
        return dirAndPdf.w * rtxState.environmentProb;
    }
    else {
        if (rnd < rtxState.environmentProb + (1.0 - rtxState.environmentProb) * lightBufInfo.trigSampProb) {
            // Sample triangle mesh light
            return (1.0 - rtxState.environmentProb) * SampleTriangleLight(pos, lightSample) * lightBufInfo.trigSampProb;
        }
        else {
            // Sample point light
            return (1.0 - rtxState.environmentProb) * SamplePuncLight(pos, lightSample) * (1.0 - lightBufInfo.trigSampProb);
        }
    }
}

float SampleDirectLight(State state, out vec3 radiance, out vec3 dir) {
    LightSample lsample;
    float pdf = SampleDirectLightNoVisibility(state.position, lsample);
    if (IsPdfInvalid(pdf)) {
        return InvalidPdf;
    }

    Ray shadowRay;
    shadowRay.origin = OffsetRay(state.position, state.ffnormal);
    shadowRay.direction = lsample.wi;

    if (Occlusion(shadowRay, state, lsample.dist)) {
        return InvalidPdf;
    }
    radiance = lsample.Li;
    dir = lsample.wi;
    return pdf;
}

vec3 DirectLight(State state, vec3 wo) { // importance sample on light sources
    LightSample lightSample;
    float pdf = SampleDirectLightNoVisibility(state.position, lightSample);
    if (IsPdfInvalid(pdf)) {
        return vec3(0.0);
    }

    Ray shadowRay;
    shadowRay.origin = OffsetRay(state.position, state.ffnormal);
    shadowRay.direction = lightSample.wi;

    if (Occlusion(shadowRay, state, lightSample.dist)) {
        return vec3(0.0);
    }
    return lightSample.Li * Eval(state, wo, state.ffnormal, lightSample.wi, dummyPdf) *
        max(dot(state.ffnormal, lightSample.wi), 0.0) / pdf;
}

vec3 clampRadiance(vec3 radiance) {
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z)) {
        return vec3(0.0);
    }

    float lum = luminance(radiance);
    if (lum > rtxState.fireflyClampThreshold) {
        radiance *= rtxState.fireflyClampThreshold / lum;
    }
    return radiance;
}

void loadLastGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth) {
    uvec2 gInfo = imageLoad(lastGbuffer, imageCoords).xy;
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
}

void loadLastGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth, out uint matHash) {
    uvec4 gInfo = imageLoad(lastGbuffer, imageCoords);
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
    matHash = gInfo.w & 0xFF000000;
}

void loadThisGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth) {
    uvec2 gInfo = imageLoad(thisGbuffer, imageCoords).xy;
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
}

void loadThisGeometryInfo(ivec2 imageCoords, out vec3 normal, out float depth, out uint matHash) {
    uvec4 gInfo = imageLoad(thisGbuffer, imageCoords);
    normal = decompress_unit_vec(gInfo.y);
    depth = uintBitsToFloat(gInfo.x);
    matHash = gInfo.w & 0xFF000000;
}

Ray raySpawn(ivec2 coord, ivec2 sizeImage) {
    // Compute sampling position between [-1 .. 1]
    const vec2 pixelCenter = vec2(coord) + 0.5;
    const vec2 inUV = pixelCenter / vec2(sizeImage.xy);
    vec2 d = inUV * 2.0 - 1.0;
    // Compute ray origin and direction
    vec4 origin = sceneCamera.viewInverse * vec4(0, 0, 0, 1);
    vec4 target = sceneCamera.projInverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = sceneCamera.viewInverse * vec4(normalize(target.xyz), 0);
    return Ray(origin.xyz, normalize(direction.xyz));
}

vec3 getCameraPos(ivec2 coord, float dist) {
    Ray ray = raySpawn(coord, rtxState.size);
    return ray.origin + ray.direction * dist;
}

//bool getDirectStateFromGBuffer(uimage2D gBuffer, Ray ray, out State state, out float depth) {
//    uvec4 gInfo = imageLoad(gBuffer, imageCoords);
//    depth = uintBitsToFloat(gInfo.x);
//    if (depth >= INFINITY * 0.8)
//        return false;
//    state.position = ray.origin + ray.direction * depth;
//    state.normal = decompress_unit_vec(gInfo.y);
//    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;
//
//    state.mat.albedo = unpackUnorm4x8(gInfo.w).xyz;
//    vec4 matInfo = unpackUnorm4x8(gInfo.z);
//    state.mat.metallic = matInfo.x;
//    state.mat.roughness = matInfo.y;
//    state.mat.ior = matInfo.z * MAX_IOR_MINUS_ONE + 1.f;
//    state.mat.transmission = matInfo.w;
//    state.matID = gInfo.w >> 24;
//    return true;
//}

//# define FETCH_GEOM_CHECK_4_SUBPIXELS 0
//
//bool getIndirectStateFromGBuffer(uimage2D gBuffer, Ray ray, out State state, out float depth) {
//#if !FETCH_GEOM_CHECK_4_SUBPIXELS
//    uvec4 gInfo = imageLoad(gBuffer, imageCoords);
//    depth = uintBitsToFloat(gInfo.x);
//    if (depth >= INFINITY * 0.8)
//        return false;
//    state.position = ray.origin + ray.direction * depth;
//    state.normal = decompress_unit_vec(gInfo.y);
//    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;
//
//    // Filling material structures
//    state.mat.albedo = unpackUnorm4x8(gInfo.w).xyz;
//    vec4 matInfo = unpackUnorm4x8(gInfo.z);
//    state.mat.metallic = matInfo.x;
//    state.mat.roughness = matInfo.y;
//    state.mat.ior = matInfo.z * MAX_IOR_MINUS_ONE + 1.f;
//    state.mat.transmission = matInfo.w;
//    state.matID = gInfo.w >> 24; // hashed matarial id
//#else
//    uvec4 gInfo00 = imageLoad(gBuffer, imageCoords + ivec2(0, 0));
//    uvec4 gInfo10 = imageLoad(gBuffer, imageCoords + ivec2(1, 0));
//    uvec4 gInfo11 = imageLoad(gBuffer, imageCoords + ivec2(1, 1));
//    uvec4 gInfo01 = imageLoad(gBuffer, imageCoords + ivec2(0, 1));
//
//    depth = (uintBitsToFloat(gInfo00.x) + uintBitsToFloat(gInfo10.x) + uintBitsToFloat(gInfo11.x) +
//        uintBitsToFloat(gInfo01.x)) * 0.25;
//
//    if (depth >= INFINITY - EPS * 10.0)
//        return false;
//
//    state.position = ray.origin + ray.direction * depth;
//    state.normal = (decompress_unit_vec(gInfo00.y) + decompress_unit_vec(gInfo10.y) + decompress_unit_vec(gInfo11.y) +
//        decompress_unit_vec(gInfo01.y)) * 0.25;
//    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;
//
//    // Filling material structures
//    state.mat.albedo = (unpackUnorm4x8(gInfo00.w).xyz + unpackUnorm4x8(gInfo10.w).xyz +
//        unpackUnorm4x8(gInfo11.w).xyz + unpackUnorm4x8(gInfo01.w).xyz) * 0.25;
//
//    vec4 matInfo00 = unpackUnorm4x8(gInfo00.z);
//    vec4 matInfo10 = unpackUnorm4x8(gInfo10.z);
//    vec4 matInfo11 = unpackUnorm4x8(gInfo11.z);
//    vec4 matInfo01 = unpackUnorm4x8(gInfo01.z);
//
//    state.mat.metallic = (matInfo00.x + matInfo10.x + matInfo11.x + matInfo01.x) * 0.25;
//    state.mat.roughness = (matInfo00.y + matInfo10.y + matInfo11.y + matInfo01.y) * 0.25;
//    state.mat.ior = (matInfo00.z + matInfo10.z + matInfo11.z + matInfo01.z) * 0.25 * MAX_IOR_MINUS_ONE + 1.f;
//    state.mat.transmission = (matInfo00.w + matInfo01.w + matInfo11.w + matInfo10.w) * 0.25;
//
//    float r = rand(prd.seed);
//    if (r < 0.25) {
//        state.matID = gInfo00.w >> 24;
//    }
//    else if (r < 0.5) {
//        state.matID = gInfo10.w >> 24;
//    }
//    else if (r < 0.75) {
//        state.matID = gInfo11.w >> 24;
//    }
//    else {
//        state.matID = gInfo01.w >> 24;
//    }
//#endif
//    return true;
//}

vec3 DebugInfo(in State state) {
    switch (rtxState.debugging_mode) {
    case eMetallic:
        return vec3(state.mat.metallic);
    case eNormal:
        return (state.normal + vec3(1)) * .5;
    case eDepth:
        return vec3(0.0);
    case eBaseColor:
        return state.mat.albedo;
    case eEmissive:
        return state.mat.emission;
    case eAlpha:
        return vec3(state.mat.alpha);
    case eRoughness:
        return vec3(state.mat.roughness);
    case eTexcoord:
        return vec3(state.texCoord, 0);
    };
    return vec3(1000, 0, 0);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PathTrace_MetallicWorkflow(Ray r)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    vec3 absorption = vec3(0.0);

    float samplePdf;
    vec3 sampleWi;
    vec3 sampleBSDF;
    Material lastMaterial;
    for (int depth = 0; depth < rtxState.maxDepth; depth++)
    {
        ClosestHit(r);

        // Hitting the environment
        if (prd.hitT == INFINITY)
        {
            if (rtxState.debugging_mode != eNoDebug)
            {
                if (depth != rtxState.maxDepth - 1)
                    return vec3(0);
                if (rtxState.debugging_mode == eRadiance)
                    return radiance;
                else if (rtxState.debugging_mode == eWeight)
                    return throughput;
                else if (rtxState.debugging_mode == eRayDir)
                    return (r.direction + vec3(1)) * 0.5;
            }

            vec3 env;
            vec3 Li;
            float MIS_weight = 1;
            if (_sunAndSky.in_use == 1)
                env = sun_and_sky(_sunAndSky, r.direction);
            else
            {
                vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
                env = texture(environmentTexture, uv).rgb;
                Li = env;
            }
            if (depth > 0 /*&& lastMaterial.transmission == 0*/)
            {
                float lightPdf;
                Li = EnvEval(sampleWi, lightPdf);
                MIS_weight = powerHeuristic(samplePdf, lightPdf);
            }
            // Done sampling return
            return radiance + (Li * rtxState.hdrMultiplier * throughput * MIS_weight);
        }


        BsdfSampleRec bsdfSampleRec;

        // Get Position, Normal, Tangents, Texture Coordinates, Color
        ShadeState sstate = GetShadeState(prd);

        State state;
        state.position = sstate.position;
        state.normal = sstate.normal;
        state.tangent = sstate.tangent_u[0];
        state.bitangent = sstate.tangent_v[0];
        state.texCoord = sstate.text_coords[0];
        state.matID = sstate.matIndex;
        state.isEmitter = false;
        state.specularBounce = false;
        state.isSubsurface = false;
        state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;

        // Filling material structures
        GetMaterialsAndTextures(state, r);

        if (state.isEmitter) {
            float lightPdf;
            vec3 Li = LightEval(state, prd.hitT, sampleWi, lightPdf);
            float MIS_weight = ((depth == 0 /*|| lastMaterial.transmission == 0*/) ? 1.0f : powerHeuristic(samplePdf, lightPdf));
            radiance += Li * throughput * MIS_weight;
            break;
        }

        // Color at vertices
        state.mat.albedo *= sstate.color;

        // Debugging info
        if (rtxState.debugging_mode != eNoDebug && rtxState.debugging_mode < eRadiance)
        {
            // return vec3(0.0f, 1.0f, 0.0f);
            return DebugInfo(state);
        }

        vec3 wo = -r.direction;
        // MIS 
        vec3 Li, wi;
        float lightPdf = SampleDirectLight(state, Li, wi);
        if (!IsPdfInvalid(lightPdf) /*&& lastMaterial.transmission == 0*/) {
            float BSDFPdf = Pdf(state, wo, state.ffnormal, wi);
            float weight = powerHeuristic(lightPdf, BSDFPdf);
            radiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) *
                throughput / lightPdf * weight;
        }

        sampleBSDF = Sample(state, wo, state.ffnormal, sampleWi, samplePdf, prd.seed);

        if (IsPdfInvalid(samplePdf)) {
            break;
        }

        throughput *= sampleBSDF / samplePdf * absDot(state.ffnormal, sampleWi);

        r.origin = OffsetRay(state.position, state.ffnormal);
        r.direction = sampleWi;

		lastMaterial = state.mat;

        //return radiance;

        //// KHR_materials_unlit
        //if (state.mat.unlit)
        //{
        //    return radiance + state.mat.albedo * throughput;
        //}

        //// Reset absorption when ray is going out of surface
        //if (dot(state.normal, state.ffnormal) > 0.0)
        //{
        //    absorption = vec3(0.0);
        //}

        //// Emissive material
        //radiance += state.mat.emission * throughput;

        //// Add absoption (transmission / volume)
        //throughput *= exp(-absorption * prd.hitT);

        //// Light and environment contribution
        //VisibilityContribution vcontrib = DirectLight(r, state);
        //vcontrib.radiance *= throughput;

        //// Sampling for the next ray
        //bsdfSampleRec.f = Sample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf, prd.seed);

        //// Set absorption only if the ray is currently inside the object.
        //if (dot(state.ffnormal, bsdfSampleRec.L) < 0.0)
        //{
        //    absorption = -log(state.mat.attenuationColor) / vec3(state.mat.attenuationDistance);
        //}

        //if (bsdfSampleRec.pdf > 0.0)
        //{
        //    throughput *= bsdfSampleRec.f * abs(dot(state.ffnormal, bsdfSampleRec.L)) / bsdfSampleRec.pdf;
        //}
        //else
        //{
        //    break;
        //}

        //// Debugging info
        //if (rtxState.debugging_mode != eNoDebug && (depth == rtxState.maxDepth - 1))
        //{
        //    if (rtxState.debugging_mode == eRadiance)
        //        return vcontrib.radiance;
        //    else if (rtxState.debugging_mode == eWeight)
        //        return throughput;
        //    else if (rtxState.debugging_mode == eRayDir)
        //        return (bsdfSampleRec.L + vec3(1)) * 0.5;
        //}

#ifdef RR
        // For Russian-Roulette (minimizing live state)
        float rrPcont = (depth >= RR_DEPTH) ?
            min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) :
            1.0;
#endif

        // Next ray
        //r.direction = bsdfSampleRec.L;
        //r.origin = OffsetRay(sstate.position, dot(bsdfSampleRec.L, state.ffnormal) > 0 ? state.ffnormal : -state.ffnormal);

        // We are adding the contribution to the radiance only if the ray is not occluded by an object.
        // This is done here to minimize live state across ray-trace calls.
        //if (vcontrib.visible == true)
        //{
        //    // Shoot shadow ray up to the light (1e32 == environement)
        //    Ray  shadowRay = Ray(r.origin, vcontrib.lightDir);
        //    bool inShadow = AnyHit(shadowRay, vcontrib.lightDist);
        //    if (!inShadow)
        //    {
        //        radiance += vcontrib.radiance;
        //    }
        //}


#ifdef RR
        if (rand(prd.seed) >= rrPcont)
            break;                // paths with low throughput that won't contribute
        throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
    }


    return radiance;
}

vec3 PathTrace_MetallicWorkflowNoMIS(Ray r)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    vec3 absorption = vec3(0.0);

    float samplePdf;
    vec3 sampleWi;
    vec3 sampleBSDF;
    Material lastMaterial;
    for (int depth = 0; depth < rtxState.maxDepth; depth++)
    {
        ClosestHit(r);

        // Hitting the environment
        if (prd.hitT == INFINITY)
        {
            if (rtxState.debugging_mode != eNoDebug)
            {
                if (depth != rtxState.maxDepth - 1)
                    return vec3(0);
                if (rtxState.debugging_mode == eRadiance)
                    return radiance;
                else if (rtxState.debugging_mode == eWeight)
                    return throughput;
                else if (rtxState.debugging_mode == eRayDir)
                    return (r.direction + vec3(1)) * 0.5;
            }

            vec3 env;
            vec3 Li;
            float MIS_weight = 1;
            if (_sunAndSky.in_use == 1)
                env = sun_and_sky(_sunAndSky, r.direction);
            else
            {
                vec2 uv = GetSphericalUv(r.direction);  // See sampling.glsl
                env = texture(environmentTexture, uv).rgb;
                Li = env;
            }
            // Done sampling return
            return radiance + (Li * rtxState.hdrMultiplier * throughput);
        }


        BsdfSampleRec bsdfSampleRec;

        // Get Position, Normal, Tangents, Texture Coordinates, Color
        State state = GetState(prd, r.direction);

        //State state;
        //state.position = sstate.position;
        //state.normal = sstate.normal;
        //state.tangent = sstate.tangent_u[0];
        //state.bitangent = sstate.tangent_v[0];
        //state.texCoord = sstate.text_coords[0];
        //state.matID = sstate.matIndex;
        //state.isEmitter = false;
        //state.specularBounce = false;
        //state.isSubsurface = false;
        //state.ffnormal = dot(state.normal, r.direction) <= 0.0 ? state.normal : -state.normal;

        // Filling material structures
        GetMaterialsAndTextures(state, r);

        if (state.isEmitter) {
            float lightPdf;
            vec3 Li = LightEval(state, prd.hitT, sampleWi, lightPdf);
            radiance += Li * throughput;
            break;
        }

        //// Color at vertices
        //state.mat.albedo *= sstate.color;

        // Debugging info
        if (rtxState.debugging_mode != eNoDebug && rtxState.debugging_mode < eRadiance)
        {
            // return vec3(0.0f, 1.0f, 0.0f);
            return DebugInfo(state);
        }

        vec3 wo = -r.direction;

        sampleBSDF = Sample(state, wo, state.ffnormal, sampleWi, samplePdf, prd.seed);

        if (IsPdfInvalid(samplePdf)) {
            break;
        }

        throughput *= sampleBSDF / samplePdf * absDot(state.ffnormal, sampleWi);

        r.origin = OffsetRay(state.position, state.ffnormal);
        r.direction = sampleWi;

        lastMaterial = state.mat;
    }
    return radiance;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 samplePixel_MetallicWorkflow(ivec2 imageCoords, ivec2 sizeImage)
{
    vec3 pixelColor = vec3(0);

    // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
    vec2 subpixel_jitter = rtxState.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(prd.seed), rand(prd.seed));

    // Compute sampling position between [-1 .. 1]
    const vec2 pixelCenter = vec2(imageCoords) + subpixel_jitter;
    const vec2 inUV = pixelCenter / vec2(sizeImage.xy);
    vec2       d = inUV * 2.0 - 1.0;

    // Compute ray origin and direction
    vec4 origin = sceneCamera.viewInverse * vec4(0, 0, 0, 1);
    vec4 target = sceneCamera.projInverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = sceneCamera.viewInverse * vec4(normalize(target.xyz), 0);

    // Depth-of-Field
    vec3  focalPoint = sceneCamera.focalDist * direction.xyz;
    float cam_r1 = rand(prd.seed) * M_TWO_PI;
    float cam_r2 = rand(prd.seed) * sceneCamera.aperture;
    vec4  cam_right = sceneCamera.viewInverse * vec4(1, 0, 0, 0);
    vec4  cam_up = sceneCamera.viewInverse * vec4(0, 1, 0, 0);
    vec3  randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
    vec3  finalRayDir = normalize(focalPoint - randomAperturePos);

    Ray ray = Ray(origin.xyz + randomAperturePos, finalRayDir);


    vec3 radiance = PathTrace_MetallicWorkflow(ray);

    // Removing fireflies
    float lum = dot(radiance, vec3(0.212671f, 0.715160f, 0.072169f));
    if (lum > rtxState.fireflyClampThreshold)
    {
        radiance *= rtxState.fireflyClampThreshold / lum;
    }

    return radiance;
}