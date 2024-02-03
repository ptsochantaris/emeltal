#include <metal_stdlib>

using namespace metal;

// Based on "sinebow" from Inferno
[[ stitchable ]] half4 genie(float2 position, half4 color, float2 size, float time) {
    half widthScale = sin((position.y / size.y) * M_PI_F);

    position.x -= size.x * 0.5;
    half2 uv = half2(position / 50);
    half wave = sin(uv.y + time) * widthScale;

    half3 waveColor = half3(color);

    for (half i = 0.0h; i < 10.0h; i++) {
        half x = sin(uv.y * sin(time) + i * 0.2h + time);
        uv.x += 0.08h * x * widthScale;

        half luma = abs(1.0h / (400.0h * uv.x + wave * 200)) * widthScale;
        half e = sin(i * 0.3h + time * 0.9) * 0.5h + 0.5h;
        waveColor = max(waveColor, waveColor + e * luma);
    }

    return half4(waveColor, color.a);
}

float map(float3 p, float iTime) {
    const half k = 1.0;
    float d = 2.0;
    for (int i = 0; i < 16; i++) {
        half fi = half(i);
        half time = iTime * (fract(fi * 412.531 + 0.513) - 0.5) * 2.0;
        float3 P = p + sin(time + fi * float3(52.5126, 64.62744, 632.25)) * float3(2.0, 2.0, 0.8);
        half S = mix(0.5, 1.0, fract(fi * 412.531 + 0.5124));

        float L = length(P)-S;
        float h = clamp( 0.5 + 0.5*(d-L)/k, 0.0, 1.0 );
        d = mix( d, L, h ) - k*h*(1.0-h);
    }
    return d;
}

// Adapted from https://www.shadertoy.com/view/3sySRK
[[ stitchable ]] half4 modelBackground(float2 position, half4 color, float2 size, float time) {
    const float3 rayDir = float3(0.0, 0.0, -1.0);
    const float h = 1e-5; // or some other value
    const float2 k = float2(1,-1);
    const float2 uv = position/size.xy;

    // screen size is 6m x 6m
    float3 rayOri = float3((uv - 0.5) * float2(size.x/size.y, 1.0) * 7.0, 4.0);

    half depth = 0.0;
    float3 p;
    float t = time / 4.0;

    for(int i = 0; i < 16; i++) {
        p = rayOri + rayDir * depth;
        half dist = map(p, t);
        depth += dist;
        if (dist < 1e-6) {
            break;
        }
    }

    float3 n = normalize( k.xyy*map( p + k.xyy*h, t ) +
                      k.yyx*map( p + k.yyx*h, t ) +
                      k.yxy*map( p + k.yxy*h, t ) +
                      k.xxx*map( p + k.xxx*h, t ) );

    half b = max(0.0, dot(n, float3(0.577)));
    half c = 2.0 - ((depth - 0.5) / 2.0);
    float col = ((0.5 + 0.5 * cos(b * 2.0)) * (0.90 + b * 0.10)) * exp( -depth * 0.005 ) * c;
    float C = max(0.0, col * 0.2);
    return half4(C, C, C, 1);
}
