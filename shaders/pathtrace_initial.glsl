#define ENVMAP 1
#define RR 1        // Using russian roulette
#define RR_DEPTH 2  // Minimum depth

#include "pbr_metallicworkflow.glsl"
#include "gltf_material.glsl"
#include "env_sampling.glsl"
#include "shade_state.glsl"
#include "punctual.glsl"
#include "HashBuildStructure.glsl"

float dummyPdf;

bool IsPdfInvalid(float p) {
    return p <= 1e-8 || isnan(p);
}

bool Occlusion(Ray ray, State state, float dist) {
    return AnyHit(ray, dist - abs(ray.origin.x - state.position.x) -
        abs(ray.origin.y - state.position.y) -
        abs(ray.origin.z - state.position.z) - 1);
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

bool getStateFromGbuffer(uimage2D gBuffer, Ray ray, ivec2 imageCoords, out State state, out float depth)
{
    uvec4 gInfo = imageLoad(gBuffer, imageCoords);
    depth = uintBitsToFloat(gInfo.x);
    if (depth >= INFINITY * 0.8)
        return false;
    state.position = ray.origin + ray.direction * depth;
    state.normal = decompress_unit_vec(gInfo.y);
    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;

    state.mat.albedo = unpackUnorm4x8(gInfo.w).xyz;
    vec4 matInfo = unpackUnorm4x8(gInfo.z);
    state.mat.metallic = matInfo.x;
    state.mat.roughness = matInfo.y;
    state.mat.ior = matInfo.z * MAX_IOR_MINUS_ONE + 1.f;
    state.mat.transmission = matInfo.w;
    state.matID = gInfo.w >> 24;
    return true;
}

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

float MIS(float f, float g) {
    return powerHeuristic(f, g);
}

# define FETCH_GEOM_CHECK_4_SUBPIXELS 0
bool getIndirectStateFromGBuffer(uimage2D gBuffer, Ray ray, ivec2 imageCoords, out State state, out float depth) {
#if !FETCH_GEOM_CHECK_4_SUBPIXELS
    uvec4 gInfo = imageLoad(gBuffer, imageCoords);
    depth = uintBitsToFloat(gInfo.x);
    if (depth >= INFINITY * 0.8)
        return false;
    state.position = ray.origin + ray.direction * depth;
    state.normal = decompress_unit_vec(gInfo.y);
    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;

    // Filling material structures
    state.mat.albedo = unpackUnorm4x8(gInfo.w).xyz;
    vec4 matInfo = unpackUnorm4x8(gInfo.z);
    state.mat.metallic = matInfo.x;
    state.mat.roughness = matInfo.y;
    state.mat.ior = matInfo.z * MAX_IOR_MINUS_ONE + 1.f;
    state.mat.transmission = matInfo.w;
    state.matID = gInfo.w >> 24; // hashed matarial id
#else
    uvec4 gInfo00 = imageLoad(gBuffer, imageCoords + ivec2(0, 0));
    uvec4 gInfo10 = imageLoad(gBuffer, imageCoords + ivec2(1, 0));
    uvec4 gInfo11 = imageLoad(gBuffer, imageCoords + ivec2(1, 1));
    uvec4 gInfo01 = imageLoad(gBuffer, imageCoords + ivec2(0, 1));

    depth = (uintBitsToFloat(gInfo00.x) + uintBitsToFloat(gInfo10.x) + uintBitsToFloat(gInfo11.x) +
        uintBitsToFloat(gInfo01.x)) * 0.25;

    if (depth >= INFINITY - EPS * 10.0)
        return false;

    state.position = ray.origin + ray.direction * depth;
    state.normal = (decompress_unit_vec(gInfo00.y) + decompress_unit_vec(gInfo10.y) + decompress_unit_vec(gInfo11.y) +
        decompress_unit_vec(gInfo01.y)) * 0.25;
    state.ffnormal = dot(state.normal, ray.direction) <= 0.0 ? state.normal : -state.normal;

    // Filling material structures
    state.mat.albedo = (unpackUnorm4x8(gInfo00.w).xyz + unpackUnorm4x8(gInfo10.w).xyz +
        unpackUnorm4x8(gInfo11.w).xyz + unpackUnorm4x8(gInfo01.w).xyz) * 0.25;

    vec4 matInfo00 = unpackUnorm4x8(gInfo00.z);
    vec4 matInfo10 = unpackUnorm4x8(gInfo10.z);
    vec4 matInfo11 = unpackUnorm4x8(gInfo11.z);
    vec4 matInfo01 = unpackUnorm4x8(gInfo01.z);

    state.mat.metallic = (matInfo00.x + matInfo10.x + matInfo11.x + matInfo01.x) * 0.25;
    state.mat.roughness = (matInfo00.y + matInfo10.y + matInfo11.y + matInfo01.y) * 0.25;
    state.mat.ior = (matInfo00.z + matInfo10.z + matInfo11.z + matInfo01.z) * 0.25 * MAX_IOR_MINUS_ONE + 1.f;
    state.mat.transmission = (matInfo00.w + matInfo01.w + matInfo11.w + matInfo10.w) * 0.25;

    float r = rand(prd.seed);
    if (r < 0.25) {
        state.matID = gInfo00.w >> 24;
    }
    else if (r < 0.5) {
        state.matID = gInfo10.w >> 24;
    }
    else if (r < 0.75) {
        state.matID = gInfo11.w >> 24;
    }
    else {
        state.matID = gInfo01.w >> 24;
    }
#endif
    return true;
}

// Pack the hit info to a uvec4
// Depth 32bit, Normal 32bit, Metallic 8bit, Roughness 8bit, IOR 8bit, Transmission 8bit, Albedo 24bit, Hashed Material ID 8bit
uvec4 encodeGeometryInfo(State state, float depth) {
    uvec4 gInfo;
    // Depth
    gInfo.x = floatBitsToUint(depth);

    gInfo.y = compress_unit_vec(state.ffnormal);
    gInfo.z = packUnorm4x8(vec4(state.mat.metallic, state.mat.roughness, (state.mat.ior - 1.0) / MAX_IOR_MINUS_ONE, state.mat.transmission));
    gInfo.w = packUnorm4x8(vec4(state.mat.albedo, 1.0)) & 0xFFFFFF; //agbr
    gInfo.w += hash8bit(state.matID);
    return gInfo;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 PathTrace_Initial(Ray r, inout PathPayLoad pathState)
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
        pathState.currentVertexIndex++;
        bool isPrimaryHit = (pathState.currentVertexIndex == 1);
        ClosestHit(r);

        BsdfSampleRec bsdfSampleRec;

        // Get Position, Normal, Tangents, Texture Coordinates, Color
        ShadeState sstate;
        State state;

        // If hit the scene, get the state
        if (prd.hitT != INFINITY)
        {
            sstate = GetShadeState(prd);

            //// Debug the world position
            //return sstate.position;

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
        }

        // Transfer normal to a uint
        uint normprint = BinaryNorm(state.ffnormal);

        // for debugging
        //return vec3(float(cellIndex

        // Flag shows if it is the reconnect vertex
        bool connectabele = (pathState.isLastVertexClassifiedAsRough > 0) && (pathState.currentVertexIndex == pathState.rcVertexLength);

        if (connectabele)
        {
            // If hit the scene, record the vertex position and normal
            if (prd.hitT != INFINITY)
            {
                pathState.rcVertexPos = state.position;
                pathState.rcVertexNorm = state.ffnormal;
            }
            else
            {
                pathState.rcEnvDir = r.direction;
                pathState.rcEnv = 1;
                pathState.rcVertexPos = state.position + r.direction * INFINITY * 0.8;
                pathState.rcVertexNorm = -r.direction;
            }
            pathState.thp = vec3(1.0f);
        }

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
            float lightPdf;
            Li = EnvEval(r.direction, lightPdf);
            if (depth > 0 /*&& lastMaterial.transmission == 0*/)
            {
                MIS_weight = powerHeuristic(samplePdf, lightPdf);
            }
            // Done sampling return
            pathState.radiance += (Li * rtxState.hdrMultiplier * throughput * MIS_weight);

            if (pathState.currentVertexIndex < pathState.rcVertexLength)
            {
                vec3 thp = pathState.prefixThp;
                pathState.prefixPathRadiance += thp * rtxState.hdrMultiplier * Li * MIS_weight;
            }
            else
            {
                vec3 thp = pathState.thp;
                pathState.rcVertexRadiance += thp * rtxState.hdrMultiplier * Li * MIS_weight;
            }

            return radiance + (Li * rtxState.hdrMultiplier * throughput * MIS_weight);
        }

        if (state.isEmitter) {
            float lightPdf;
            vec3 Li = LightEval(state, prd.hitT, sampleWi, lightPdf);
            float MIS_weight = ((depth == 0 /*|| lastMaterial.transmission == 0*/) ? 1.0f : powerHeuristic(samplePdf, lightPdf));

            pathState.radiance += Li * throughput * MIS_weight;
            if (pathState.currentVertexIndex < pathState.rcVertexLength)
            {
                vec3 thp = pathState.prefixThp;
                pathState.prefixPathRadiance += thp * Li * MIS_weight;
            }
            else
            {
                vec3 thp = pathState.thp;
                pathState.rcVertexRadiance += thp * Li * MIS_weight;
            }

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

        // Do MIS 
        {
            // Direct light radiance and direction from sample point to light sample
            vec3 Li, wi;

            // Sample direct light
            float lightPdf = SampleDirectLight(state, Li, wi);

            // If the light is visible, do MIS
            if (!IsPdfInvalid(lightPdf) /*&& lastMaterial.transmission == 0*/) {

                // Get material BSDF and pdf according to w_out, w_in and  face normal
                float BSDFPdf = Pdf(state, wo, state.ffnormal, wi);

                // Evaluate MIS weight
                float MIS_weight = powerHeuristic(lightPdf, BSDFPdf);

                // Accumulate path radiance
                // Sample from a direct light source, contribution = throughput * Li * bsdf * cos(theta) / pdf_light * MIS_weight
                pathState.radiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) *
                    throughput / lightPdf * MIS_weight;

                // update reconnect radiance/prefix point radiance
                if (pathState.currentVertexIndex < pathState.rcVertexLength)
                {
                    // Get the prefix throughput
                    vec3 thp = pathState.prefixThp;
                    pathState.prefixPathRadiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) *
                        thp / lightPdf * MIS_weight;
                }
                else
                {
                    // Get the reconnect throughput
                    vec3 thp = pathState.thp;
                    pathState.rcVertexRadiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) *
                        thp / lightPdf * MIS_weight;
                }

                // Accumulate total radiance
                radiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) *
                    throughput / lightPdf * MIS_weight;
            }
        }

		if (pathState.currentVertexIndex < pathState.rcVertexLength)
		{
            // Pack the hit info of the prev vertex
            pathState.preRcVertexHitInfo = encodeGeometryInfo(state, prd.hitT);

            // Record the output direction of the prev vertex(start from the prev vertex), it will be used in bsdf eval
            pathState.preRcVertexWo = wo;

            // Record the position of the prev vertex
            pathState.preRcVertexPos = state.position;

            // Record the face normal of the prev vertex
            pathState.preRcVertexNorm = state.ffnormal;

            // Record mat ID of prev vertex
            pathState.preRcMatId = int(state.matID);
		}

        // Sample Next Ray
        {
            // Sample next ray according to BSDF
            sampleBSDF = Sample(state, wo, state.ffnormal, sampleWi, samplePdf, prd.seed);

            // Vlidate sample
            if (IsPdfInvalid(samplePdf)) {
                break;
            }

            // Get roughness
            float matRoughness = state.mat.roughness;

            // TODO
            // Set roughness threshold, now fixed, should be a variable 
            float kRoughnessThreshold = roughnessThreshold;

            // Flag shows if rough enough to reconnect
            bool vertexClassifiedAsRough = matRoughness > kRoughnessThreshold;

            // Track last vertex roughness
            pathState.isLastVertexClassifiedAsRough = uint(vertexClassifiedAsRough);

            // If the vertex is classified as rough and do not have a reconnect vertex
            if (pathState.currentVertexIndex < pathState.rcVertexLength && vertexClassifiedAsRough)
            {
                pathState.validRcPath = 1;

                // Current vertex is prev vertex and next vertex is the reconnect vertex
                pathState.rcVertexLength = pathState.currentVertexIndex + 1;

                // Record the pdf from the prev vertex to the reconnect vertex
                pathState.pdf = samplePdf;
            }

            // evaluate the bsdf * cos(theta) / pdf
            vec3 bsdfCosWeight = sampleBSDF / samplePdf * absDot(state.ffnormal, sampleWi);

            // If the vertex is in the prefix path
            if (pathState.currentVertexIndex + 1 < pathState.rcVertexLength)
            {
                // Update prefix throughput
                pathState.prefixThp *= bsdfCosWeight;
            }
            // If the vertex is in the reconnect path
            else
            {
                // Accumulate reconnect throughput
                pathState.thp *= bsdfCosWeight;
            }

            // Cache the bsdfCosWeight from the prev vertex to reconnect vertex
            if (pathState.currentVertexIndex + 1 == pathState.rcVertexLength)
            {
                pathState.cacheBsdfCosWeight = bsdfCosWeight;
                // pathState.cacheBsdfCosWeight = sampleBSDF;
            }

            throughput *= bsdfCosWeight;

            r.origin = OffsetRay(state.position, state.ffnormal);
            r.direction = sampleWi;

            lastMaterial = state.mat;
        }

        // Russian Roulette
        {
#if RR
            // If next vertex is the reconnect vertex, do not terminate the path
            bool nextReconnect = (pathState.isLastVertexClassifiedAsRough > 0) && (pathState.currentVertexIndex + 1 == pathState.rcVertexLength);
            // For Russian-Roulette (minimizing live state)
            float rrPcont = ((depth >= RR_DEPTH) && !nextReconnect) ?
                min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) :
                1.0;
            if (rand(prd.seed) >= rrPcont)
            {
                break;                // paths with low throughput that won't contribute
            }
            throughput /= rrPcont;  // boost the energy of the non-terminated paths

            // If the vertex is in the prefix path
            if (pathState.currentVertexIndex + 1 < pathState.rcVertexLength)
            {
                // Update prefix throughput
                pathState.prefixThp /= rrPcont;
            }
            // If the vertex is in the reconnect path
            else
            {
                // Accumulate reconnect throughput
                pathState.thp /= rrPcont;
            }
#endif
        }
    }

    return radiance;
}

//------------------------------------------------------------
//
float pHatIndirect(vec3 L)
{
    return Luminance(L);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void PathTrace_initial_Indirect(State state, Ray ray, inout PathPayLoad pathState, ivec2 imageCoords)
{
    // state is the first hit point's state
    pathState.preRcVertexHitInfo = encodeGeometryInfo(state, 0);
    // Define some basic variables
    vec3 throughput = vec3(/* multiBounce ? 4.0 :*/ 1.0);
    pathState.preRcVertexWo = -ray.direction;

    // Modulate for denoise
    // state.mat.albedo = vec3(1.0);

    for (int depth = 1; depth <= rtxState.maxDepth; depth++)
    {
        vec3 wo = -ray.direction;

        // MIS
        if (depth > 1)
        {
            vec3 Li, wi;
            float lightPdf = SampleDirectLight(state, Li, wi);

            if (!IsPdfInvalid(lightPdf))
            {
                float BSDFPdf = Pdf(state, wo, state.ffnormal, wi);
                float weight = MIS(lightPdf, BSDFPdf);
                pathState.rcVertexRadiance += Li * BSDF(state, wo, state.ffnormal, wi) * absDot(state.ffnormal, wi) * throughput / lightPdf * weight;
            }
        }

        // Sample next ray
        vec3 sampleWi;
        float samplePdf;
        vec3 sampleBSDF = Sample(state, wo, state.ffnormal, sampleWi, samplePdf, prd.seed);

        if (depth == 1)
        {
            pathState.pdf = samplePdf;
            pathState.preRcVertexPos = state.position;
            pathState.preRcVertexNorm = state.ffnormal;
            pathState.preRcMatId = int(state.matID);
        }

        if (IsPdfInvalid(samplePdf))
        {
            break;
        }

        if (depth > 1)
        {
            // Accumulate throughputs
            throughput *= sampleBSDF / samplePdf * absDot(state.ffnormal, sampleWi);
        }
        else
        {
            // Record the first hit
        }

        // Generate next ray
        ray.origin = OffsetRay(state.position, state.ffnormal);
        ray.direction = sampleWi;

        ClosestHit(ray);

        // Hit Env
        if (prd.hitT >= INFINITY)
        {
            if (depth > 1)
            {
                float lightPdf;
                vec3 Li = EnvEval(sampleWi, lightPdf);
                float weight = MIS(samplePdf, lightPdf);
                pathState.rcVertexRadiance += Li * throughput * weight;
            }
            else
            {
                // Hack here. To let samples from the infinity light able to be temporally reused
                // giSample.matInfo = uvec2(0xFFFFFFFF, 0xFFFFFFFF);

                // Update the Reconnect point
                pathState.rcEnvDir = ray.direction;
                pathState.rcEnv = 1;
                pathState.rcVertexPos = state.position + sampleWi * INFINITY * 0.8;
                pathState.rcVertexNorm = -sampleWi;
            }
            break;
        }

        state = GetState(prd, ray.direction);
        GetMaterials(state, ray);
		state.matID = int(hash8bit(state.matID));

        // Hit Light
        if (state.isEmitter)
        {
            if (depth > 1)
            {

                float lightPdf;
                vec3 Li = LightEval(state, prd.hitT, sampleWi, lightPdf);
                float weight = MIS(samplePdf, lightPdf);
                pathState.rcVertexRadiance += Li * throughput * weight;
            }
            else
            {
                // Record the Reconnect point
                pathState.rcVertexPos = state.position;
                pathState.rcVertexNorm = state.ffnormal;
            }
            break;
        }

        if (depth == 1)
        {
            // Record the first hit
            pathState.rcVertexPos = state.position;
            pathState.rcVertexNorm = state.ffnormal;
        }

#ifndef RR
        float rrPcont = (1 >= RR_DEPTH) ? min(max(throughput.x, max(throughput.y, throughput.z)) * state.eta * state.eta + 0.001, 0.95) : 1.0;
        if (rand(prd.seed) >= rrPcont) {
            break;
        }
        throughput /= rrPcont;
#endif
    }
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 samplePixel_Initial(ivec2 imageCoords, ivec2 sizeImage, uint idx)
{
    vec3 pixelColor = vec3(0);

    Ray ray = raySpawn(imageCoords, ivec2(sizeImage));
    // imageStore(debugUintImage, imageCoords, ivec4(prd.seed));

    PathPayLoad pathState;
    pathState.preRcVertexHitInfo = uvec4(0);
    pathState.currentVertexIndex = 0;
    pathState.thp = vec3(1.0f, 1.0f, 1.0f);
    pathState.prefixThp = vec3(1.0f, 1.0f, 1.0f);
    pathState.radiance = vec3(0.0f, 0.0f, 0.0f);
    pathState.prefixPathRadiance = vec3(0.0f, 0.0f, 0.0f);
    pathState.rcVertexRadiance = vec3(0.0f, 0.0f, 0.0f);
    pathState.rcEnv = 0;
    pathState.rcEnvDir = vec3(0.0f, 0.0f, 0.0f);
    pathState.pdf = 0.0f;
    pathState.cacheBsdfCosWeight = vec3(0);

    pathState.rcVertexPos = vec3(0.0f);
    pathState.rcVertexNorm = vec3(100.f);
    pathState.preRcVertexPos = vec3(0.0f);
    pathState.preRcVertexNorm = vec3(100.0f);
    pathState.preRcMatId = -1;

    pathState.validRcPath = 0;

    // rcVertexLength is the number of vertices in the reconnect path, should be initialized to the max depth + 1
    pathState.rcVertexLength = rtxState.maxDepth + 1;

    State state;
    float depth;
    if (!getIndirectStateFromGBuffer(thisGbuffer, ray, imageCoords, state, depth))
    {
		reconnectionDataBuffer[idx].preRcVertexHitInfo = uvec4(0);
        // return vec3(0);
    }

    state.position += state.ffnormal * 2e-2;
    // PathTrace_initial_Indirect(state, ray, pathState, imageCoords);
    vec3 radiance = vec3(0);
    radiance = PathTrace_Initial(ray, pathState);

    reconnectionDataBuffer[idx].pathPreThp = pathState.prefixThp;
    reconnectionDataBuffer[idx].pathLength = pathState.rcVertexLength;
    reconnectionDataBuffer[idx].pathPreRadiance = pathState.prefixPathRadiance;
    reconnectionDataBuffer[idx].preRcVertexWo = pathState.preRcVertexWo;
    reconnectionDataBuffer[idx].preRcVertexHitInfo = pathState.preRcVertexHitInfo;

    // Assign initialSample
    initialSampleBuffer[idx].rcVertexLo = pathState.rcVertexRadiance;
    initialSampleBuffer[idx].rcVertexPos = pathState.rcVertexPos;
    initialSampleBuffer[idx].rcVertexNorm = pathState.rcVertexNorm;
    initialSampleBuffer[idx].preRcVertexPos = pathState.preRcVertexPos;
    initialSampleBuffer[idx].preRcVertexNorm = pathState.preRcVertexNorm;
    initialSampleBuffer[idx].pdf = pathState.pdf;
    initialSampleBuffer[idx].rcEnv = pathState.rcEnv;
    initialSampleBuffer[idx].rcEnvDir = pathState.rcEnvDir;
	initialSampleBuffer[idx].preRcMatId = pathState.preRcMatId;
    initialSampleBuffer[idx].pHat = pHatIndirect(pathState.rcVertexRadiance);

    // Removing fireflies
    float lum = dot(radiance, vec3(0.212671f, 0.715160f, 0.072169f));
    if (lum > rtxState.fireflyClampThreshold)
    {
        pathState.radiance *= rtxState.fireflyClampThreshold / lum;
        radiance *= rtxState.fireflyClampThreshold / lum;
    }

    // Removing fireflies
    lum = dot(pathState.prefixPathRadiance, vec3(0.212671f, 0.715160f, 0.072169f));
    if (lum > rtxState.fireflyClampThreshold)
    {
        pathState.prefixPathRadiance *= rtxState.fireflyClampThreshold / lum;
    }

    // Removing fireflies
    lum = dot(pathState.rcVertexRadiance, vec3(0.212671f, 0.715160f, 0.072169f));
    if (lum > rtxState.fireflyClampThreshold)
    {
        pathState.rcVertexRadiance *= rtxState.fireflyClampThreshold / lum;
    }


    // return radiance;
    return pathState.prefixPathRadiance + pathState.rcVertexRadiance * pathState.cacheBsdfCosWeight * pathState.prefixThp;
}