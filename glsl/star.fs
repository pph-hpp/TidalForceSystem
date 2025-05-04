#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
// texture sampler
uniform sampler2D texture1;


void main()
{
    vec4 texColor = texture(texture1, TexCoord);
    FragColor = vec4(texColor.rgb, 1.0);  // 仅传递 RGB 通道
    // FragColor = texture(texture1, TexCoord);
}