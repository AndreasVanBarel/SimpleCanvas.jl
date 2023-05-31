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

close(c)
# note that m can still be used here.
```

# Features 

### Custom color map

Change the color map by calling `setcolormap(c, cmap)`. This function `cmap` should map values in `c` (whatever type they may have) to a `Vector{UInt8}` of 3 elements representing the red, green and blue channels. The default function attempts to interpret the contents of `c` as real numbers, and maps `0.0` to `UInt8[0,0,0]` and `1.0` to `UInt8[255,255,255]`. E.g.,

```julia
function cmap(value::Float64)
    if value < 0.0; r = g = b = 0.0;
    elseif value <= 0.25; r = 0.0; g = 4 * value; b = 1.0; 
    elseif value <= 0.5; r = 0.0; g = 1.0; b = 1.0 - 4 * (value - 0.25); 
    elseif value <= 0.75; r = 4 * (value - 0.5); g = 1.0; b = 0.0
    elseif value <= 1.0; r = 1.0; g = 1.0 - 4 * (value - 0.75); b = 0.0
    else; r = g = b = 0.0; end
    round.(UInt8, [255r, 255g, 255b])
end
setcolormap(c, cmap) # c::Canvas
```

# Notes

- Do not edit the fields of c directly, i.e., use provided functions such as `setcolormap` instead of editing `c.colormap`.
- Support for multiple simultaneous canvases is experimental.
- Currently linear indexing of a canvas is not yet supported, i.e., `c[1] = 1` does not work.


