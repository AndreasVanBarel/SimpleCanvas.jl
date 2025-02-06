# SimpleCanvas.jl
A simple canvas for drawing to the screen pixel by pixel. A `Matrix` is wrapped in the custom type `Canvas` that subtypes `AbstractMatrix`. 
One can read and write from that canvas as if it were any other `Matrix`, but the contents are shown in real time on the screen. 
This happens partially asynchronously, such that the canvas can still be used somewhat efficiently in calculations.
One could pass the canvas to any routine that allows `AbstractMatrix`, and see in real time how that routine updates the matrix. 
This could be educational for understanding certain linear algebra algorithms. 

## Installation

By using the [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/):

```julia
] add https://github.com/AndreasVanBarel/SimpleCanvas.jl.git
```

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
```

# Features 

### Custom color map

Change the color map by calling `colormap!(c, cmap)`. This function `cmap` should map values in `c` (whatever type they may have) to a `Tuple{UInt8, UInt8, UInt8}` representing the red, green and blue channels. The default function attempts to interpret the contents of `c` as real numbers, and maps `0.0` to `(0,0,0)` and `1.0` to `(255,255,255)`. E.g.,

```julia
function colormap(value::Float64)
    if value < 0.0; r = g = b = 0.0;
    elseif value <= 0.25; r = 0.0; g = 4 * value; b = 1.0; 
    elseif value <= 0.5; r = 0.0; g = 1.0; b = 1.0 - 4 * (value - 0.25); 
    elseif value <= 0.75; r = 4 * (value - 0.5); g = 1.0; b = 0.0;
    elseif value <= 1.0; r = 1.0; g = 1.0 - 4 * (value - 0.75); b = 0.0;
    else; r = g = b = 0.0; end
    round.(UInt8, 255 .*(r,g,b))
end
colormap!(c, cmap) # c::Canvas
```

### Other options

Given a `Canvas c`:
 - `name!(c,n)` changes the window name to `n`.
 - `windowsize!(c,w,h)` sets the window size to `w` by `h`.
 - `windowscale!(c,scale=1)` sets the window size to the size of the underlying canvas, scaled by `scale`.
 - `target_fps!(c,t)` sets target fps to `t`.
 - `show_fps!(c,true)` shows measured fps in window title. (Available in a future release)
 - `diagnostic_level!(c,lvl)` shows diagnostic messages. `lvl = 0`: none, `lvl = 1`: only uploads to GPU, `lvl = 2`: also background redraws 

# Notes

- Do not edit the fields of a `Canvas` directly, i.e., use provided functions such as `colormap!` instead of editing `c.colormap`.
- Support for multiple simultaneous canvases is experimental.


