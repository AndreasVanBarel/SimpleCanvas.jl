include("Vec2ds.jl")
using .Vec2ds

mutable struct Sprite
	texture::UInt32 #Location of the texture on the GPU
	width::UInt32
	height::UInt32
	isrgba::Bool
	vertices::Vector{Vec2d}
end

function Sprite(p1::Vec2d,p2::Vec2d,p3::Vec2d,p4::Vec2d,tex::Array{UInt8,3})
	vertices = [crosspoint(p1,p2,p3,p4),p1,p2,p3,p4]

	#Store and configure texture on the GPU
	textureP = UInt32[0]
	glGenTextures(1, textureP)
	to_gpu(tex, textureP)

    width, height = size(tex)[2:3]
    isrgba = size(tex)[1]==4
	Sprite(textureP[1],width,height,isrgba,vertices)
end
const default_vertices = [Vec2d(-1.0,1.0), Vec2d(1.0,1.0), Vec2d(-1.0,-1.0), Vec2d(1.0,-1.0)]
# const default_vertices = [Vec2d(-1.0,-1.0), Vec2d(-1.0,1.0), Vec2d(1.0,-1.0), Vec2d(1.0,1.0)]
Sprite(tex::Array{UInt8,3}) = Sprite(default_vertices..., tex)

# Construct Sprite with existing texture pointer
function Sprite(p1::Vec2d,p2::Vec2d,p3::Vec2d,p4::Vec2d,textureP::Integer)
	# width and height of existing texture can be found as follows:
	glBindTexture(GL_TEXTURE_2D, textureP)
	w = UInt[0]; h = UInt[0]; αsize = UInt[0]; 
	miplevel = 0
	glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_WIDTH, w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_HEIGHT, h);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, miplevel, GL_TEXTURE_ALPHA_SIZE, αsize);
	width = w[1]; height = h[1]; isrgba = αsize[1] > 0

	vertices = [crosspoint(p1,p2,p3,p4),p1,p2,p3,p4]
	Sprite(textureP,width,height,isrgba,vertices)
end
Sprite(textureP::Integer) = Sprite(default_vertices..., textureP)

# Calculates the center of mass of the given 4 vertices
# (previously calculated the intersection between p1--p3 and p2--p4)
function crosspoint(p1::Vec2d,p3::Vec2d,p2::Vec2d,p4::Vec2d)
	# A = [p2.y-p1.y	p1.x-p2.x;
	# 	 p4.y-p3.y	p3.x-p4.x]
	# b = [p2.y*p1.x-p2.x*p1.y, p4.y*p3.x-p4.x*p3.y]
	# p = A\b
	# Vec2d(p...)
	(p1+p2+p3+p4)/4
end
Sprite(p::Vec2d,v1::Vec2d,v2::Vec2d,tex) = Sprite(p,p+v1,p+v1+v2,p+v2,tex)
Sprite(p::Vec2d,tex) = Sprite(p,VEC_EX,VEC_EY,tex)
Sprite(tex) = Sprite(VEC_ORIGIN,tex)
free(s::Sprite) = glDeleteTextures(1,[s.texture])
loc(s::Sprite) = s.vertices[2]
center(s::Sprite) = s.vertices[1]
shape!(s::Sprite,p1::Vec2d,p2::Vec2d,p3::Vec2d,p4::Vec2d) = s.vertices .= [crosspoint(p1,p2,p3,p4),p1,p2,p3,p4]
shape!(s::Sprite,p::Vec2d,v1::Vec2d,v2::Vec2d) = shape!(s,p,p+v1,p+v1+v2,p+v2)
function draw(s::Sprite)
	glBindVertexArray(sprite_vao)
	glBindBuffer(GL_ARRAY_BUFFER, sprite_vbo)
	c,p1,p2,p3,p4 = s.vertices
	# data = [c.x,c.y,0f0,0.5f0,0.5f0,
	# 		p1.x,p1.y,0f0, 1f0,0f0,
	# 		p2.x,p2.y,0f0, 1f0,1f0,
	# 		p3.x,p3.y,0f0, 0f0,1f0,
	# 		p4.x,p4.y,0f0, 0f0,0f0,
	# 		p1.x,p1.y,0f0, 1f0,0f0]
	data = [c.x,c.y,0f0,0.5f0,0.5f0,
			p1.x,p1.y,0f0, 0f0,0f0,
			p2.x,p2.y,0f0, 0f0,1f0,
			p4.x,p4.y,0f0, 1f0,1f0,
			p3.x,p3.y,0f0, 1f0,0f0,
			p1.x,p1.y,0f0, 0f0,0f0]
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW) #copy to the GPU
	glUseProgram(programs.sprite)
	glUniform4f(programs.sprite_colorLoc, 1f0, 1f0, 1f0, 0.5f0)
	glActiveTexture(GL_TEXTURE0)
	glBindTexture(GL_TEXTURE_2D, s.texture)
	glUniform1i(programs.sprite_textureLoc, 0) #NOTE: This 0 corresponds to the GL_TEXTURE0 that is active.
	glDrawArrays(GL_TRIANGLE_FAN, 0, 6)
end
Base.show(s::Sprite,io::IO) = println("Sprite with texture stored on GPU at $(s.texture)")