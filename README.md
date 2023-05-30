# SimpleCanvas.jl
A simple canvas for drawing to the screen pixel by pixel. A `Matrix` is wrapped in the custom type `Canvas` that subtypes `AbstractMatrix`. 
One can read and write from that canvas as if it were any other `Matrix`, but the contents are shown in real time on the screen. 
This happens partially asynchronously, such that the canvas can still be used somewhat efficiently in calculations.
One could pass the canvas to any routine that allows `AbstractMatrix`, and see in real time how that routine updates the matrix. 
This could be educational for understanding certain linear algebra algorithms. 

!! Still a work in progress !!

## Installation

This is installed using the standard tools of the [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/):

```julia
pkg> add https://github.com/AndreasVanBarel/SimpleCanvas.jl.git
```
You get the `pkg>` prompt by hitting `]` as the first character of the line.

# Example

```julia
m = zeros(600,800);
c = canvas(m); # c can be used as if it were a Matrix

# Drawing random rectangles
for i = 1:10000
    i%100 == 0 && println("drawing $i")
    x,y = rand(1:600-50), rand(1:800-50)
    c[x:x+50, y:y+50].= 1 .-c[x:x+50, y:y+50]
end

# close(c)
# note that m can still be used here.
```
