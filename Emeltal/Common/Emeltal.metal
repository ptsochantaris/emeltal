#include <metal_stdlib>

using namespace metal;

///////////////////////////////// Genie

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

//////////////////////////////////// Blobs

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

////////////////////////////////// Net

// Adapted from https://www.shadertoy.com/view/lscczl

#define S(a, b, t) smoothstep(a, b, t)

float N21(float2 p) {
    float3 a = fract(float3(p.xyx) * float3(213.897, 653.453, 253.098));
    a += dot(a, a.yzx + 79.76);
    return fract((a.x + a.y) * a.z);
}

float2 GetPos(float2 id, float2 offs, float t) {
    float n = N21(id+offs);
    float n1 = fract(n*10.);
    float n2 = fract(n*100.);
    float a = t+n;
    return offs + float2(sin(a*n1), cos(a*n2))*.4;
}

float df_line(float2 a, float2 b, float2 p) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba) / dot(ba,ba), 0., 1.);
    return length(pa - ba * h);
}

float line(float2 a, float2 b, float2 uv) {
    float r1 = .04;
    float r2 = .01;

    float d = df_line(a, b, uv);
    float d2 = length(a-b);
    float fade = S(1.3, .5, d2);

    fade += S(.05, .02, abs(d2-.75));
    return S(r1, r2, d)*fade;
}

[[ stitchable ]] half4 pickerBackground(float2 position, half4 color, float2 size, float time) {
    float2 uv = (position-size.xy*.5)/size.y;
    float2 st = 8.*uv;
    float2 id = floor(st);

    st = fract(st)-.5;

    float2 p[9];
    int i=0;
    for(float y=-1.; y<=1.; y++) {
        for(float x=-1.; x<=1.; x++) {
            p[i++] = GetPos(id, float2(x,y), time);
        }
    }

    float m = 0.;
    float sparkle = 0.;

    for(int i=0; i<9; i++) {
        m += line(p[4], p[i], st);

        float d = length(st-p[i]);

        float s = (.002/(d*d));
        s *= S(1., .7, d);
        float pulse = sin((fract(p[i].x)+fract(p[i].y)+time)*5.)*.4+.6;
        pulse = pow(pulse, 20.);

        s *= pulse;
        sparkle += s;
    }

    m += line(p[1], p[3], st);
    m += line(p[1], p[5], st);
    m += line(p[7], p[5], st);
    m += line(p[7], p[3], st);
    m += sparkle;

    half D = dot(uv,uv);
    half3 col = half3(0.05 * m);

    return half4(col + ((D/2) + 0.5) * half3(0.4, 0.7, 0.9), 1);
}
