shader_type canvas_item;

uniform sampler2D noiseTexture : repeat_enable;
uniform float speed : hint_range(0.01,1.0) = 0.1; 
uniform float amplitude : hint_range(0.01,1.0) = 0.2; 
uniform float aliasing = 0.01;

float circle(vec2 center, float radius, vec2 p){
	return length(p - center) - radius;
}

void fragment(){
	vec4 noise;
	noise = texture(noiseTexture, UV + speed * (TIME)) - 0.5;
	
	float sdf = circle(vec2(0.5, 0.5), 0.4, UV) + amplitude * noise.x;
	
	COLOR.rgb = vec3(1.0);
	if (sdf > 0.0)
        COLOR = vec4(0.0, 0.0, 0.0, 0.0);
    else
		COLOR.a = smoothstep(sdf, sdf + aliasing, 0.0);
}

