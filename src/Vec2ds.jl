module Vec2ds

export Vec2d 
export VEC_ORIGIN, VEC_EX, VEC_EY
export p_to_n, n_to_p

import Base: +,-,*,/,rand
import LinearAlgebra: norm, dot, inv

struct Vec2d
    x::Float32
    y::Float32
end
+(a::Vec2d,b::Vec2d) = Vec2d(a.x+b.x,a.y+b.y)
-(a::Vec2d,b::Vec2d) = Vec2d(a.x-b.x,a.y-b.y)
-(a::Vec2d) = Vec2d(-a.x,-a.y)
*(c,a::Vec2d) = Vec2d(c*a.x,c*a.y)
*(a::Vec2d,c) = *(c,a)
/(a::Vec2d,c) = Vec2d(a.x/c,a.y/c)
norm(v::Vec2d) = sqrt(v.x*v.x+v.y*v.y)
dot(a::Vec2d,b::Vec2d) = a.x*b.x + a.y*b.y
dist(a::Vec2d,b::Vec2d) = norm(a-b)
const VEC_ORIGIN = Vec2d(0f0,0f0)
const VEC_EX = Vec2d(1f0,0f0)
const VEC_EY = Vec2d(0f0,1f0)

p_to_n(p::Vec2d) = Vec2d(2p.x/width()-1, -2p.y/height()+1) #pixel coordinates p to normalized device coordinates
n_to_p(n::Vec2d) = Vec2d((n.x+1)*width(), (1-n.y)*height()) #normalized device coordinates n to pixel coordinates

rand(::Type{Vec2d},args...) = Vec2d.(rand(args...).*2 .-1, rand(args...).*2 .-1)

end