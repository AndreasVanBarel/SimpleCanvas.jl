# SimpleCanvas.jl
A simple canvas for drawing to the screen pixel by pixel. A `Matrix` is wrapped in the custom type `Canvas` that subtypes `AbstractMatrix`. 
One can read and write from that canvas as if it were any other `Matrix`, but the contents are shown in real time on the screen. 
This happens partially asynchronously, such that the canvas can still be used somewhat efficiently in calculations.
One could pass the canvas to any function that takes an `AbstractMatrix` parameter, and, see in real time how that function updates the matrix. 
This could be educational for understanding certain linear algebra algorithms, e.g., in place `qr!` or `lu!`.

Support for MacOSX coming in a future update.

## Installation

By using the [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/):

```julia
] add https://github.com/AndreasVanBarel/SimpleCanvas.jl.git
```

# Example

```julia
using SimpleCanvas

m = zeros(600,800);
c = canvas(m); # c can be used as if it were a matrix

# draw random rectangles
for i = 1:10000
    i%100 == 0 && println("drawing $i")
    x,y = rand(1:600-50), rand(1:800-50)
    c[x:x+50, y:y+50] .= 1 .- c[x:x+50, y:y+50]
end

# close(c) # or manually close the window
```

# Features 

### Custom color map

Change the color map by calling `colormap!(c, cmap)`. The function `cmap` should map a value in `c` (whatever type it may have) to a `Tuple{UInt8, UInt8, UInt8}` representing the red, green and blue channels. E.g.,

```julia
w, h = 800, 600
start, stop = -0.2, 1.2
c = canvas(ones(h).*LinRange(start, stop, w)') # h×w Canvas{Float64}
function cmap_blue_to_red(value::Float64)
    if value < 0.0; r = g = 0.0; b = 1.0;
    elseif value <= 0.25; r = 0.0; g = 4 * value; b = 1.0; 
    elseif value <= 0.5; r = 0.0; g = 1.0; b = 1.0 - 4 * (value - 0.25); 
    elseif value <= 0.75; r = 4 * (value - 0.5); g = 1.0; b = 0.0;
    elseif value <= 1.0; r = 1.0; g = 1.0 - 4 * (value - 0.75); b = 0.0;
    else; r = 1.0; g = b = 0.0; end
    round.(UInt8, 255 .*(r,g,b))
end
colormap!(c, cmap_blue_to_red)
```

### Other options

Given a `Canvas c`:
 - `name!(c,n)` changes the window name to `n`.
 - `windowsize!(c,w,h)` sets the window size to `w` by `h`.
 - `windowscale!(c,scale=1)` sets the window size to the size of the underlying canvas, scaled by `scale`.
 - `target_fps!(c,t)` sets target fps to `t`.
 - `show_fps!(c,true)` shows measured fps in window title.

# Notes

- Do not edit the fields of a `Canvas` directly. E.g., use `colormap!` instead of editing `c.colormap`.
- Support for multiple simultaneous canvases is experimental.


