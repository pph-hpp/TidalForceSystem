#version 330 core

in vec3 textCoords;
out vec4 FragColor;

vec3 saturnColor(float u, float r) {
    // 颜色分区
    vec3 equatorColor = vec3(225.0, 190.0, 130.0) / 255.0;
    vec3 bandColor    = vec3(255.0, 210.0, 150.0) / 255.0;
    vec3 polarColor   = vec3(180.0, 140.0, 110.0) / 255.0;

    // 纬度带状条纹：通过 u 和 r 控制带纹理
    float bands = sin(10.0 * u + r * 20.0) * 0.5 + 0.5;
    vec3 latColor = mix(equatorColor, bandColor, bands);

    // 极区渐变（越接近极点越灰）
    float polarFactor = abs(cos(u)); // u=0,π -> 极点；u=π/2 -> 赤道
    vec3 finalColor = mix(latColor, polarColor, polarFactor);

    return finalColor;
}


// HSV to RGB 转换
vec3 hsv2rgb(float h, float s, float v) {
    float c = v * s;
    float h6 = h * 6.0;
    float x = c * (1.0 - abs(mod(h6, 2.0) - 1.0));
    vec3 rgb;

    if (0.0 <= h6 && h6 < 1.0)
        rgb = vec3(c, x, 0.0);
    else if (1.0 <= h6 && h6 < 2.0)
        rgb = vec3(x, c, 0.0);
    else if (2.0 <= h6 && h6 < 3.0)
        rgb = vec3(0.0, c, x);
    else if (3.0 <= h6 && h6 < 4.0)
        rgb = vec3(0.0, x, c);
    else if (4.0 <= h6 && h6 < 5.0)
        rgb = vec3(x, 0.0, c);
    else
        rgb = vec3(c, 0.0, x);

    float m = v - c;
    return rgb + vec3(m);
}

vec3 magmaColor(float r, float u, float v) {
    // 模拟流动岩浆的动态纹理（可加时间变量增强流动感）
    float noise = sin(15.0 * u + 12.0 * v + r * 25.0) * 0.5 + 0.5;
    vec3 core = mix(vec3(1.0, 0.3, 0.0), vec3(1.0, 1.0, 0.0), noise); // 橙到黄
    core = mix(core, vec3(0.2, 0.0, 0.0), pow(1.0 - r, 3.0)); // 越深越暗红
    return core;
}



void main(){
    float r = textCoords.x;
    float u = textCoords.y;
    float v = textCoords.z;

    // 判断是否为“内部”
    vec3 color;

    // if (r >= 0.7) {
    //     // 表面：土星纹理
    //     color = saturnColor(u, r);
    // } else {
    //     // 内部：岩浆纹理
    //     color = magmaColor(r, u, v);
    // }
    // vec3 equatorColor = vec3(225.0, 190.0, 130.0) / 255.0;
    // FragColor = vec4(equatorColor, 1.0);

    float azimuth = v;   // v ∈ [0, 2π]
    float zenith  = u;

    if (azimuth >= 0.6 * 3.1415926 && azimuth <= 1.6 * 3.1415926) {
        // ?? 昼半球，渲染为木星色调
        float hue = 0.08 + 0.05 * sin(10.0 * zenith);  // 褐黄纹理
        float sat = 0.6;
        float val = 0.75;
        color = hsv2rgb(hue, sat, val);
    } else {
        // ? 夜半球，渲染为地球蓝绿色
        float hue = 0.55 + 0.05 * sin(3.0 * zenith);
        float sat = 0.6;
        float val = 0.8;
        color = hsv2rgb(hue, sat, val);
    }
    FragColor = vec4(color, 1.0);
}