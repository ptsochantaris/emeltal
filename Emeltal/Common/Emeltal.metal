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

float2 GetGradient(float2 intPos, float t) {
    float rand = fract(sin(dot(intPos, float2(12.9898, 78.233))) * 43758.5453);
    float angle = 6.283185 * rand + 4.0 * t * rand;
    return float2(cos(angle), sin(angle));
}

[[ stitchable ]] half4 modelBackground(float2 position, half4 color, float2 size, float time) {
    float2 uv = position / size.y;
    float3 pos = float3(uv * 2, time * 0.2);

    float2 i = floor(pos.xy);
    float2 f = pos.xy - i;
    float2 blend = f * f * (3.0 - 2.0 * f);
    float noiseVal =
        mix(
            mix(
                dot(GetGradient(i + float2(0, 0), pos.z), f - float2(0, 0)),
                dot(GetGradient(i + float2(1, 0), pos.z), f - float2(1, 0)),
                blend.x),
            mix(
                dot(GetGradient(i + float2(0, 1), pos.z), f - float2(0, 1)),
                dot(GetGradient(i + float2(1, 1), pos.z), f - float2(1, 1)),
                blend.x),
        blend.y
    );

    half wave = abs(sin(position.y));
    half noise = noiseVal * wave * 0.4;
    return half4(color.x - noise, color.y - noise, color.z - noise, color.a);
}
