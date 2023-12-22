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
