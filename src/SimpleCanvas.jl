"""
	SimpleCanvas

A module for drawing to the screen pixel by pixel.

# Exports
- `canvas(M::Matrix)`: Creates a `Canvas` for the matrix `M`.
- `close(C::Canvas)`: Closes `C`, releasing its resources.
- `colormap!(C::Canvas, colormap::Function)`: Sets the color mapping function.
- `name!(C::Canvas, name)`: Sets the window name.
- `window_size!(C::Canvas, width, height)`: Sets the window size.
- `window_scale!(C::Canvas, scale)`: Scales the window.
- `target_fps!(C::Canvas, fps)`: Sets the target frames per second.
- `actual_fps(C::Canvas): Returns the actual frames per second.

# Examples
```julia
using SimpleCanvas
C = canvas(zeros(600,800)); # C can be used as if it were a Matrix
C[101:200, 201:300] .= 1
```
"""
module SimpleCanvas

export Canvas
export canvas, close, colormap!, name!, window_size!, window_scale!
export target_fps!, show_fps!, actual_fps

# Exported temporarily for development purposes
export to_gpu, map_to_rgb!, diagnostic_level!
export colormap_grayscale, colormap_spy, colormap_identity, colormap_rgb
export colormap_wheel, colormap_blue_to_red

using GLFW
using ModernGL
using LinearAlgebra

using Base.Threads

import Base: close
import Base: show
import Base: size, getindex, setindex!

include("Sprite.jl")
include("GLPrograms.jl")
import .GLPrograms

# Settings
default_fps = 60
default_diagnostic_level = 0
default_show_fps = true
default_updates_immediately = true
default_name = "Simple Canvas"

# Developer Settings
max_time_fraction = 0.99
const rgb_immediately = true # for performance purposes made constant
# note that setting this to false requires the use of a colormap that is defined before the canvas is created
# Debugging
const DEBUGGING = true # for performance purposes made constant
debug(x...) = DEBUGGING && println("DEBUG: ", x...)

programs = nothing


"""
	Canvas{T} <: AbstractMatrix{T}

A canvas for drawing to the screen pixel by pixel. The canvas is backed by a matrix of type `T` and can be used as if it were a matrix. For constructing a `Canvas`, see `canvas`.
"""
mutable struct Canvas{T} <: AbstractMatrix{T}
	# Core functionality
    m::Matrix{T}
	rgb::Array{UInt8, 3} # RGB matrix
	colormap::Function # T -> UInt8[3] 
    window
	sprite

	# Tasks
	polling_task
	drawing_task
	drawing_cond # Condition for waking up the drawing task

	# Timing and update management
    last_update::UInt64 # last time that update!() was called (time at the start of the call)
	next_update::UInt64 # earliest next time that a call may yield to the drawing task
	update_pending::Bool # whether there is a change to m that is not uploaded to the GPU and thus not reflected on the canvas
	updates_immediately::Bool
	fps::Number # target frames per second under load
	E_Δt_iteration::Float64 # exponential moving average of the time between iterations

	# Additional functionality
	show_fps::Bool
	diagnostic_level::Int # 0: none, 1: uploads to GPU, 2: + redraws
	name::String

    # finalizing (releasing memory when canvas becomes inaccessible)
    function Canvas{T}(args...) where T
        C = new(args...)
        finalizer(close, C)
    end
end

get_width(C::Canvas) = size(C.m)[2]
get_height(C::Canvas) = size(C.m)[1]

# TODO: Create a better construction interface, probably with Canvas type itself as name
# such that Canvas{T}(w,h) creates a canvas of size w×h
"""
	canvas(M::AbstractMatrix; name)
	canvas(M::AbstractMatrix, width, height; name)

Create a `Canvas{T}` initialized by the contents of `AbstractMatrix{T}` `M`. The contents of `M` are copied and can be discarded afterwards.
The canvas window will have the same size as `M`, unless `width` and `height` are specified explicitly.

# Examples
```julia
using SimpleCanvas
w,h = 800, 600
M = zeros(h,w);
C = canvas(M); # C can be used as if it were a Matrix
# M is independent of C here. Write in C to update the canvas.
```
"""
function canvas(m::AbstractMatrix{T}, width::Integer, height::Integer; name::String=default_name) where T 
	debug("Creating canvas for matrix on thread $(Threads.threadid())")
	m = collect(m)::Matrix{T} # Ensures that the matrix is copied and stored in a contiguous block of memory
	rgb = Array{UInt8, 3}(undef, 3, size(m)[1], size(m)[2])
	drawing_cond = Threads.Condition()
	cmap = default_colormap(T)
    C = Canvas{T}(m, rgb, cmap, nothing, nothing,
		nothing, nothing, drawing_cond,
		0, 0, true, default_updates_immediately, default_fps, 1/default_fps,
		default_show_fps, default_diagnostic_level, name)
		
	# Create a window with an OpenGL context
	C.window = GLFW.CreateWindow(width, height, name)
	isnothing(C.window) && @error("Window or OpenGL context creation failed.")
	configure_window(C)
	configure_context(C)

	map_to_rgb!(C) # Initializes C.rgb
	C.sprite = Sprite(C.rgb) 
	detach_context() # Detach the OpenGL context from this thread
	
	# Create the polling task
	# Note: If we don't want the spawned Task to keep the canvas alive, we could use a WeakRef:
	# Cref = WeakRef(C) # This reference does not prevent the canvas from being garbage collected
	C.polling_task = @task polling_task(C)
	schedule(C.polling_task)
	C.drawing_task = Threads.@spawn :interactive drawing_task(C)
    return C
end

canvas(m::AbstractMatrix{T}) where T = canvas(m, size(m)[2], size(m)[1]; name=default_name)

"""
	canvas(T::Type, width, height; name)

Equivalent to `canvas(zeros(T,height,width), width, height; name=name)`.
"""
canvas(T::Type, w::Integer, h::Integer; name::String=default_name) = canvas(zeros(T,h,w), w, h; name=name)


###################
# Initializations #
###################

# Initializes GLFW library
function init_glfw()
	GLFW.Init() || @error("GLFW failed to initialize")

	# Specify OpenGL version (Note: MacOSX supports at most OpenGL 4.1)
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3);
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3);
	GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE);
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE); # ? Supposedly needed for MacOSX
end
init_glfw()

function reset_glfw()
    GLFW.Terminate()
    init_glfw()
end

# Detach the current OpenGL context from the calling thread
function detach_context()
	GLFW.MakeContextCurrent(GLFW.Window(C_NULL))
end

# Configures the GLFW window and sets the callback functions
# This does not require the openGL context 
function configure_window(C::Canvas)
	window = C.window

	# Callback functions
	error_callback(x...) = @error("$(C.name): GLFW Error callback:\n$x")
	key_callback(window, key, scancode, action, mods) = debug("$(C.name): Key action: $(key), $(scancode), $(action), $(mods)")
	mouse_button_callback(window, button, action, mods) = debug("$(C.name): Mouse button action: $(button), $(action), $(mods)")
	GLFW.SetErrorCallback(error_callback)
	GLFW.SetKeyCallback(window, key_callback)
	GLFW.SetMouseButtonCallback(window, mouse_button_callback)
	GLFW.SetFramebufferSizeCallback(window, make_framebuffer_size_callback(C))
end

# Configures the OpenGL context associated to the given window
# Requires that the context is available on the calling thread
function configure_context(C::Canvas)
	window = C.window
	width, height = get_width(C), get_height(C)

	GLFW.MakeContextCurrent(window) # Make the window's context current

	glViewport(0, 0, width, height)
	GLFW.SwapInterval(0) #Activate V-sync (not necessary since we are using sleep in the polling task)

    # Shader programs
	global programs = GLPrograms.generatePrograms() # Compile the shader programs
	glUseProgram(programs.sprite); glUniformMatrix3fv(programs.sprite_camLoc, 1, false, Matrix{Float32}(I,3,3)); # Set Camera values on the gpu

	gpu_allocate() # Allocate buffers on the GPU	
end

# TODO: Should probably move to Sprite.jl. It requires an OpenGL context though.
sprite_vao = UInt32(0)
sprite_vbo = UInt32(0)
function gpu_allocate()
	### sprite ###
	vaoP = UInt32[0] # Vertex Array Object
	glGenVertexArrays(1, vaoP)
	glBindVertexArray(vaoP[1])

	vboP = UInt32[0] # Vertex Buffer Object
	glGenBuffers(1,vboP)
	glBindBuffer(GL_ARRAY_BUFFER, vboP[1])

	#Specifiy the interpretation of the data
	glEnableVertexAttribArray(0)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(0))
	glEnableVertexAttribArray(1)
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(Float32), Ptr{Nothing}(3*sizeof(Float32)))

	global sprite_vao = vaoP[1]
	global sprite_vbo = vboP[1]
end

"""
	close(C::Canvas)

Closes the canvas `C`, releasing its resources.
"""
function close(C::Canvas)
	C.updates_immediately = false # Prevents unnecessary yields to a drawing task that will close anyway
	free(C.sprite)
	isnothing(C.window) && return
    GLFW.SetWindowShouldClose(C.window, true)
    GLFW.DestroyWindow(C.window)
	C.window = nothing
    return
end

function make_framebuffer_size_callback(C::Canvas)
	function framebuffer_size_callback(window::GLFW.Window, width, height)
		debug("$(C.name) was resized to $width × $height")
		C.update_pending = true
		return
	end
	return framebuffer_size_callback
end


###############################
# General functions on Canvas #
###############################

function show(io::IO, C::Canvas) 
    println(io, "$(typeof(C)) named '$(C.name)'")
    # show(io, C.m)
end

function show(io::IO, mime::MIME"text/plain", C::Canvas)
    println(io, "$(typeof(C)) named '$(C.name)'")
    # show(io, mime, C.m)
end

"""
	actual_fps(C::Canvas)

Returns the actual frames per second of `C`, rounded to the nearest `Int`. Note that `C` does not update when it isn't drawn to, so the actual frames per second will be lower than the target frames per second, and tends to `0` when `C` is idle.
"""
function actual_fps(C::Canvas)
	(time_ns() - C.last_update) < 5e9/C.fps || return 0
	E_Δt = C.E_Δt_iteration
	return round(Int, E_Δt == 0 ? C.fps : 1/E_Δt)
	# Δupdate = C.next_update - C.last_update
	# return round(Int, Δupdate == 0 ? C.fps : 1e9/Δupdate)
end


#########
# Tasks #
#########

# Notes:
# Two tasks are needed:
# 1. Main thread task (named polling_task)
# 	 All GLFW related tasks, such as window creation, handling, polling OS events etc 
#    Lightweight
# 2. Separate thread task (named drawing_task)
#    All openGL calls etc
#    Does the copying of the matrix to the GPU

# The reason is that GLFW window and OS related tasks must be done on the main thread, unless one is willing to sacrifice robustness and compatibility.

function polling_task(C::Canvas)
	debug("started polling task for $(C.name) on thread $(Threads.threadid())")
	try 
		while !isnothing(C.window) && !GLFW.WindowShouldClose(C.window)
			GLFW.PollEvents()
			if C.show_fps
				fps = actual_fps(C)
				GLFW.SetWindowTitle(C.window, "$(C.name) - $(fps) fps")
			else
				GLFW.SetWindowTitle(C.window, C.name)
			end
			sleep(1/C.fps)
		end
	catch e 
		@error("Polling task for $(C.name) closed by error:\n$e")
	finally
		debug("polling task closing $(C.name)")
		close(C)
	end
	debug("polling task for $(C.name) finished")
end

function drawing_task(C::Canvas)
	debug("started drawing task for $(C.name) on thread $(Threads.threadid())")

	init_glfw()
	sleep(0.1) # Give the main thread some time to create the window and release the OpenGL context
	timer = nothing 

	t_last_iteration = time() - 1/C.fps # Sets the first guess for Δt_iteration to 1/C.fps
	α = 0.9 # Exponential moving average coefficient for Δt_iteration

	try
		while !isnothing(C.window) && !GLFW.WindowShouldClose(C.window)
			Δt_iteration = time() - t_last_iteration
			C.update_pending ? (C.E_Δt_iteration = α*C.E_Δt_iteration + (1-α)*Δt_iteration) : nothing
			t_last_iteration = time()

			update_was_pending = C.update_pending
			update_was_pending && debug("\n============= Drawing task for $(C.name) iteration start =============")
			t_start_update_draw = time_ns()
			update_draw!(C)
			Δt_update_draw = time_ns() - t_start_update_draw

			update_was_pending && debug("Drawing task for $(C.name): update_draw! in $(1e-9Δt_update_draw)s ($(round(Int,1e9/Δt_update_draw)) fps equivalent)")
		
			# In case the update took a long time, we provide the caller with some additional time to do his business. We make it so that at most max_time_fraction of the time goes to the canvas update. If the update takes longer, we allow the caller at least that amount divided by max_time_fraction. The target fps will then not be reached.
			if Δt_update_draw > 1e9/C.fps*max_time_fraction
				Δnext_update = round(UInt64, Δt_update_draw/max_time_fraction)
			else 
				Δnext_update = round(UInt64, 1e9/C.fps)
			end # for max_time_fraction < 1, Δnext_update is always at least 1e9/C.fps
			C.next_update = t_start_update_draw + Δnext_update
			t_timer = max(Δnext_update - Δt_update_draw, 0)

			update_was_pending && debug("Δnext_update for $(C.name) is $(1e-9Δnext_update)")
			update_was_pending && debug("wait timer for $(C.name) is $(1e-9t_timer)")

			t_start_idle = time_ns()
			timer = create_notify_timer(C.drawing_cond, 1e-9t_timer)
			try_wait(C.drawing_cond)
			close(timer)
			Δt_idle = time_ns() - t_start_idle

			update_was_pending && debug("The actual waiting time: $(1e-9Δt_idle), total: $(1e-9(Δt_update_draw+Δt_idle)) ($(round(Int,1e9/(Δt_update_draw+Δt_idle))) fps equivalent)")
		end
	catch e 
		@error("Drawing task for $(C.name) closed by error:\n$e")
	end
	debug("drawing task for $(C.name) finished")
end

function update_draw!(C::Canvas)		
	GLFW.MakeContextCurrent(C.window)
	if C.update_pending == true
		C.diagnostic_level >= 1 && println("Canvas '$(C.name)': upload to GPU & redraw (initiated by polling task)")
		w,h = GLFW.GetWindowSize(C.window) # get window size
		glViewport(0, 0, w, h) # In case the window was resized
		update!(C) 
	end
	err = glGetError()
	if err != 0
		throw(ErrorException("GL Error code $err"))
	end
	detach_context()
	return 
end

@inline function mark_for_update(C::Canvas)
	C.update_pending = true
	if C.updates_immediately
		t = time_ns()
		if t > C.next_update
			# println("Yielding to drawing task, t-C.next_update = $(1e-9(t-C.next_update))")
			C.diagnostic_level >= 1 && println("Canvas '$(C.name)': notified drawing task (by mark_for_update)")
			C.next_update = C.next_update + round(Int, 1e9/C.fps) # Ensures that this won't be called again until at least 1/fps seconds have passed
			# Note that the drawing task will refine C.next_update once it is clear how long the update took. This is a mere minimal time between updates.
			try_notify(C.drawing_cond)
			sleep(0.001) # This ensures a 'long' yield to the drawing task
		end
	end
end

@inline function try_notify(cond)
    lock(cond)
    try
        notify(cond)
	catch e 
		debug("Error during try_notify: $e")
    finally
        unlock(cond)
    end
end

@inline function try_wait(cond)
	lock(cond)
	try
		wait(cond)
	catch e 
		debug("Error during try_wait: $e")
	finally
		unlock(cond)
	end
end

function create_notify_timer(cond, timeout)
	return Timer(x->try_notify(cond), timeout)
end


#############
## Options ##
#############

"""
	name!(C::Canvas, name::String)

Sets the window name of `C`.
"""
function name!(C::Canvas, name::String)
	C.name = name
	GLFW.SetWindowTitle(C.window, name)
end

"""
	window_size!(C::Canvas, width::Int, height::Int)

Sets the window size to `width` × `height`.
"""
function window_size!(C::Canvas, width::Int, height::Int)
	GLFW.SetWindowSize(C.window, width, height)
end

"""
	window_scale!(C::Canvas, scale::Real = 1)

Resizes the window size such that each element of the underlying matrix is displayed using `scale²` pixels on the screen. Calling `window_scale!(C)` thus makes each pixel correspond to a single matrix element.
"""
function window_scale!(C::Canvas, scale::Real = 1)
	width = round(Int, size(C.m)[2]*scale)
	height = round(Int, size(C.m)[1]*scale)
	window_size!(C, width, height)
end

"""
	diagnostic_level!(C::Canvas, level::Int)

Sets the diagnostic level. The diagnostic level determines the amount of information printed to the console.
"""
function diagnostic_level!(C::Canvas, level::Int)
	C.diagnostic_level = level
end

"""
	target_fps!(C::Canvas, fps::Real)

Sets the target frames per second.
"""
function target_fps!(C::Canvas, fps::Real)
	C.fps = fps
end

"""
	show_fps!(C::Canvas, show::Bool=true)

Shows the frames per second in the window title of `C`.
"""
function show_fps!(C::Canvas, show::Bool=true)
	C.show_fps = show
end

# TODO: A function to rotate canvas (just changes quad vertices of course)


##################
## Colormapping ##
##################

# Broadcast colormap on matrix
function map_to_rgb!(C::Canvas{T}, colormap::Function=C.colormap) where T
	# NOTE: The colormap is passed as an explicit argument such that the Julia compiler would specialize on it! This is very important for performance.
	try 
		for i in CartesianIndices(C.m)
			v = C.m[i]
			color = UInt8.(colormap(v))
			C.rgb[1,i] = color[1]
			C.rgb[2,i] = color[2]
			C.rgb[3,i] = color[3]
		end
	catch e
		@error("Error during colormapping for $(C.name). Use `colormap!` to set a working colormap:\n$e")
	end

	# This throws an error on the GPU for unknown reasons
	# # ptr_m = pointer(C.m)
	# Threads.@threads for i in CartesianIndices(C.m)
	# 	v = C.m[i]
	# 	# lin_index = LinearIndices(C.m)[i]
	# 	# v = unsafe_load(ptr_m, lin_index)
	# 	color = colormap(v)::NTuple{3,UInt8}
	# 	C.rgb[1,i] = color[1]
	# 	C.rgb[2,i] = color[2]
	# 	C.rgb[3,i] = color[3]
	# end
end

function colormap_grayscale(v)
	v < 0 && (v = 0.0) # truncation
	v > 1 && (v = 1.0) # truncation
	r = round(UInt8, v*255)
	return (r,r,r)
end
function colormap_grayscale(v::Integer) 
	v < 0 && (v = 0) # truncation
	v > 255 && (v = 255) # truncation
	return UInt8.((v,v,v))
end

colormap_identity(v::NTuple{3, UInt8}) = return v

# Interprets an unsigned integer as an RGB value
# For UInt32 the format is 0x00RRGGBB
# For longer UInts, the format is 0x...RRGGBB
# For UInt8, 0xBB, UInt16, 0xGGBB
function colormap_rgb(v::Unsigned)
	b = UInt8(v & 0xFF)
	g = UInt8((v >> 8) & 0xFF)
	r = UInt8((v >> 16) & 0xFF)
	return (r,g,b)
end

# Very general colormap
colormap_spy(v) = isnothing(v) ? UInt8.((0,0,0)) : UInt8.((255,255,255))
colormap_spy(v::Number) = iszero(v) ? UInt8.((0,0,0)) : UInt8.((255,255,255))

# colorwheel colormap
function colormap_wheel(value::Real)
    if value < 0.0; r = g = b = 0.0;
    elseif value <= 1/6; r = 0.0; g = 6value; b = 1.0; 
    elseif value <= 2/6; r = 0.0; g = 1.0; b = 1.0 - 6value + 1; 
    elseif value <= 3/6; r = 6value - 2; g = 1.0; b = 0.0;
    elseif value <= 4/6; r = 1.0; g = 1.0 - 6value + 3; b = 0.0;
    elseif value <= 5/6; r = 1.0; g = 0.0; b = 6value - 4;
    elseif value <= 6/6; r = 1.0 - 6value + 5; g = 0.0; b = 1.0;
    else; r = g = b = 1.0; end
    round.(UInt8, 255 .*(r,g,b))
end

# blue to red colormap
function colormap_blue_to_red(value::Real)
    if value < 0.0; r = g = 0.0; b = 1.0;
    elseif value <= 0.25; r = 0.0; g = 4 * value; b = 1.0; 
    elseif value <= 0.5; r = 0.0; g = 1.0; b = 1.0 - 4 * (value - 0.25); 
    elseif value <= 0.75; r = 4 * (value - 0.5); g = 1.0; b = 0.0;
    elseif value <= 1.0; r = 1.0; g = 1.0 - 4 * (value - 0.75); b = 0.0;
    else; r = 1.0; g = b = 0.0; end
    round.(UInt8, 255 .*(r,g,b))
end

# default colormaps during construction
default_colormap(::Type{<:Any}) = colormap_spy
default_colormap(::Type{<:Real}) = colormap_grayscale
default_colormap(::Type{<:Unsigned}) = colormap_rgb
default_colormap(::Type{UInt8}) = colormap_grayscale
default_colormap(::Type{NTuple{3,UInt8}}) = colormap_identity


"""
	colormap!(C::Canvas{T}, colormap::Function)

Sets the color mapping function. `colormap` must map an element of type `T` to a `Tuple{UInt8, UInt8, UInt8}` representing the RGB values of the corresponding pixel on the canvas. `colormap` may also return values that can be converted to `Tuple{UInt8, UInt8, UInt8}` using UInt8.(), e.g., `(255,0,0)`.

# Examples
```julia
using SimpleCanvas
C = canvas(Float64, 800, 600);
colormap!(C, v::Float64 -> v==0.0 ? (0,0,0) : (255,255,255))
```
"""
function colormap!(C::Canvas, colormap::Function)
	C.colormap = colormap
	rgb_immediately && map_to_rgb!(C)
	mark_for_update(C)
end


###################################
## Matrix operations on a Canvas ##
###################################

size(C::Canvas)	= size(C.m)
getindex(C::Canvas, args...) = getindex(C.m, args...)

# Single element V
function setindex!(C::Canvas, V, I...)
    C.m[I...] = V # set matrix itself
	if rgb_immediately
		i,j = Tuple(CartesianIndices(C.m)[I...])
		_set_rgb!(C, C.colormap, V, i, j)
	end
	mark_for_update(C)
	return V # for seamless consistency with Base.setindex!
end

function _set_rgb!(C::Canvas{T}, colormap, V, i, j) where T
	V = V isa T ? V : convert(T, V)
	rgb = C.rgb
    color_V = UInt8.(colormap(V)) # should be a tuple of 3 elements
    rgb[1,i,j] = color_V[1]
    rgb[2,i,j] = color_V[2]
    rgb[3,i,j] = color_V[3]
    # rgb[1:3, cart_indices] .= UInt8.(colormap(V))
end

# Array V
function setindex!(C::Canvas, V::AbstractArray, I...)
    C.m[I...] = V # set matrix itself
	if rgb_immediately
		cart_indices = _efficient_cart_indices(C.m, I...) # this is an array of CartesianIndex
		_set_rgb!(C, C.colormap, V, cart_indices)
    end
	mark_for_update(C)
    return V # for seamless consistency with Base.setindex!
end

# Ensures that this is lazy, i.e., the CartesianIndices are computed when needed, not allocated in advance
_efficient_cart_indices(M, I...) = CartesianIndices(M)[I...] # fallback
_efficient_cart_indices(M, i::Integer, J) = CartesianIndices(M)[i:i, J]
_efficient_cart_indices(M, I, j::Integer) = CartesianIndices(M)[I, j:j]

function _set_rgb!(C::Canvas{T}, colormap, V::AbstractArray, cart_indices) where T
	rgb = C.rgb
    for (i, idx) in enumerate(cart_indices)
		V_T = V[i] isa T ? V[i] : convert(T, V[i])
        color_V = UInt8.(colormap(V_T)) # should be a tuple of 3 elements
        rgb[1, idx] = color_V[1]
        rgb[2, idx] = color_V[2]
        rgb[3, idx] = color_V[3]
    end
end


################################
## Updating GPU mirror matrix ##
################################

# Uploads texture tex to GPU at address textureP[1]
function to_gpu(tex::Array{UInt8,3}, textureP)
    size(tex)[1]!=3 && size(tex)[1]!=4 && (@error("texture format not supported"); return nothing)
    isrgba = size(tex)[1]==4
    width, height = size(tex)[2:3]

	glBindTexture(GL_TEXTURE_2D, textureP[1])
	if isrgba #rgba
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex)
	else #rgb
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex)
	end
	# glGenerateMipmap(GL_TEXTURE_2D)
	# glFinish() # supposedly blocks until GL upload finished
end

function to_gpu(C::Canvas)
	rgb_immediately || map_to_rgb!(C, C.colormap)
	GLFW.MakeContextCurrent(C.window)
	textureP = [C.sprite.texture]
	to_gpu(C.rgb, textureP) 
end

# Requires that the OpenGL context is available on the calling thread
function update!(C::Canvas)
	isnothing(C.window) && return

	C.last_update = time_ns()
	C.update_pending = false

	to_gpu(C) # pushes data to be updated to the GPU
	redraw(C) # instructs the GPU to redraw
end

# does not push anything to gpu, just redraws. Is extremely fast (often <10μs)
# Requires that the OpenGL context is available on the calling thread
function redraw(C::Canvas)
	draw(C.sprite)
	GLFW.SwapBuffers(C.window)
end

end
