"""
	SimpleCanvas

A module for drawing to the screen pixel by pixel.

# Exports
- `canvas(M::Matrix)`: Creates a `Canvas` for the matrix `M`.
- `close(C::Canvas)`: Closes `C`, releasing its resources.
- `colormap!(C::Canvas, colormap::Function)`: Sets the colormap function for `C`.
- `name!(C::Canvas, name)`: Sets the window name of `C`.
- `windowsize!(C::Canvas, width, height)`: Sets the window size of `C`.
- `windowscale!(C::Canvas, scale)`: Scales the window of `C`.
- `diagnostic_level!(C::Canvas, level)`: Sets the diagnostic level for `C`.
- `target_fps!(C::Canvas, fps)`: Sets the target frames per second for `C`.

# Examples
```julia
using SimpleCanvas
w,h = 800, 600
M = zeros(h,w);
C = canvas(M); # C can be used as if it were a Matrix
C[101:200, 201:300] .= 1
```
"""
module SimpleCanvas

export canvas, close, colormap!, name!, windowsize!, windowscale!
export diagnostic_level!, target_fps!, show_fps!

# Exported temporarily for development purposes
export Canvas, to_gpu, map_to_rgb!
# export vsync!

using GLFW
using ModernGL
using LinearAlgebra

using Base.Threads

import Base: close
import Base: show
import Base: size, length, axes, getindex, setindex!

include("Sprite.jl")
include("GLPrograms.jl")
import .GLPrograms

# Settings
default_fps = 60
default_diagnostic_level = 1
default_show_fps = true
default_updates_immediately = true
max_time_fraction = 1.0

programs = nothing

# Debugging
const DEBUGGING = false
debug(x...) = DEBUGGING && println("DEBUG: ", x...)

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

mutable struct Canvas{T} <: AbstractMatrix{T}
	# Core functionality
    m::Matrix{T}
	rgb::Array{UInt8, 3} # RGB(A) matrix (preallocated for performance)
	colormap::Function # T -> UInt8[3] 
    window
	sprite

	# Tasks
	polling_task
	drawing_task

	# Timing and update management
    last_update::UInt64 # last time that update!() was called (time at the start of the call)
	next_update::UInt64 # earliest next time that a call may yield to the drawing task
	update_pending::Bool # whether there is a change to m that is not uploaded to the GPU and thus not reflected on the canvas
	updates_immediately::Bool
	fps::Number

	# Additional functionality
	show_fps::Bool
	diagnostic_level::Int # 0: none, 1: uploads to GPU, 2: + redraws
	name::String
	# up_minx::Int # These four determine that the submatrix to be updated is contained within
	# up_maxx::Int # m[up_minx:up_maxx, up_miny:up_maxy]
	# up_miny::Int 
	# up_maxy::Int

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
canvas(m::AbstractMatrix{T}) where T = canvas(m, size(m)[2], size(m)[1]; name = "Simple Canvas")
function canvas(m::AbstractMatrix{T}, width::Integer, height::Integer; name::String) where T 
	debug("Creating canvas for matrix on thread $(Threads.threadid())")
	m = collect(m)::Matrix{T} # Ensures that the matrix is copied and stored in a contiguous block of memory
	rgb = Array{UInt8, 3}(undef, 3, size(m)[1], size(m)[2])
    C = Canvas{T}(m, rgb, colormap_grayscale, nothing, nothing, 
		nothing, nothing,
		0, 0, true, default_updates_immediately, default_fps, 
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


###################
# Initializations #
###################

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

function close(C::Canvas)
	C.updates_immediately = false # Prevents unnecessary yields to a drawing task that will close anyway
	free(C.sprite)
	isnothing(C.window) && return
    GLFW.SetWindowShouldClose(C.window, true)
    GLFW.DestroyWindow(C.window)
	C.window = nothing
    return
end

function make_framebuffer_size_callback(C)
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
    show(io, C.m)
end

function show(io::IO, mime::MIME"text/plain", C::Canvas)
    println(io, "$(typeof(C)) named '$(C.name)'")
    show(io, mime, C.m)
end

function actual_fps(C::Canvas)
	Δupdate = C.next_update - C.last_update
	return round(Int, Δupdate == 0 ? C.fps : 1e9/Δupdate)
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

# The reason is that GLFW window and OS related tasks must be done on the main thread.
# It might be possible to not do this, at the cost of robustness and compatibility.

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
		@error("Polling task for $(C.name):\n$e")
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

	try
		while !isnothing(C.window) && !GLFW.WindowShouldClose(C.window)
			update_was_pending = C.update_pending
			t = time_ns()
			update_draw(C)
			Δt = time_ns() - t
			wait_time_s = max(1/C.fps - Δt*1e-9, 0.0)
			update_was_pending && debug("Drawing task for $(C.name) took $(1e-9Δt)s, sleeping for $wait_time_s ($(round(Int,1e9/Δt)) fps equivalent)")
			sleep(wait_time_s)
		
			# In case the update took a long time, we need to provide the caller with some additional time to do his business. We make it so that at most max_time_fraction of the time goes to the canvas update. If the update takes longer, we allow the caller at least that amount divided by max_time_fraction. The target fps will then not be reached.
			if Δt > 1e9/C.fps*max_time_fraction
				C.next_update = t + round(UInt64, Δt/max_time_fraction)
			else 
				C.next_update = t + round(UInt64, 1e9/C.fps)
			end
		end
	catch e 
		@error("Drawing task for $(C.name):\n$e")
	# finally
		# println("drawing task closing $(C.name)")
		# close(C)
		# Not necessary to close the canvas; it will just remain without content, but the window should still be responsive.
	end
	debug("drawing task for $(C.name) finished")
end

function update_draw(C::Canvas)		
	GLFW.MakeContextCurrent(C.window)
	if C.update_pending == true
		C.diagnostic_level >= 1 && println("Canvas '$(C.name)': upload to GPU & redraw (initiated by polling task)")
		w,h = GLFW.GetWindowSize(C.window) # get window size
		glViewport(0, 0, w, h) # In case the window was resized
		update!(C) 
	else 
		# C.diagnostic_level >= 2 && println("Canvas '$(C.name)': redraw (initiated by polling task)")
		# redraw(C) # Redraws the canvas if no update is pending; this ensures the window is responsive
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
			C.diagnostic_level >= 1 && println("Canvas '$(C.name)': yielded")
			C.next_update = C.next_update + round(Int, 1e9/C.fps) # Ensures that this won't be called again until at least 1/fps seconds have passed
			# println("Yielding to drawing task")
			yield()
		end
	end
end


#############
## Options ##
#############

function name!(C::Canvas, name::String)
	C.name = name
	GLFW.SetWindowTitle(C.window, name)
end

function windowsize!(C::Canvas, width::Int, height::Int)
	GLFW.SetWindowSize(C.window, width, height)
end

function windowscale!(C::Canvas, scale::Real = 1)
	width = round(Int, size(C.m)[2]*scale)
	height = round(Int, size(C.m)[1]*scale)
	windowsize!(C, width, height)
end

function diagnostic_level!(C::Canvas, level::Int)
	C.diagnostic_level = level
end

function target_fps!(C::Canvas, fps::Real)
	C.fps = fps
end

function show_fps!(C::Canvas, show::Bool=true)
	C.show_fps = show
end

# This requires the OpenGL context to be available on the calling thread
# Currently not supported
# function vsync!(C::Canvas, vsync::Bool)
# 	GLFW.MakeContextCurrent(C.window)
# 	GLFW.SwapInterval(vsync ? 1 : 0)
# end


##################
## Colormapping ##
##################

# Broadcast colormap on matrix
function map_to_rgb!(C::Canvas{T}, colormap::Function=C.colormap) where T
	# NOTE: The colormap is passed as an argument such that the Julia compiler would specialize on it!

	for i in CartesianIndices(C.m)
		v = C.m[i]
		color = colormap(v)::NTuple{3,UInt8}
		C.rgb[1,i] = color[1]
		C.rgb[2,i] = color[2]
		C.rgb[3,i] = color[3]
	end
end

# A default colormap function
function colormap_grayscale(v::Float64)
	v < 0 && (v = 0.0) # truncation
	v > 1 && (v = 1.0) # truncation
	r = round(UInt8, v*255)
	return (r,r,r)
end
# Need function to convert from type T to pixel color values.
# Need function to rotate canvas (just changes quad vertices of course)

function colormap!(C::Canvas, colormap::Function)
	C.colormap = colormap
	mark_for_update(C)
end


###################################
## Matrix operations on a Canvas ##
###################################

size(C::Canvas)	= size(C.m)
length(C::Canvas) = length(C.m)	
axes(C::Canvas) = axes(C.m)	
getindex(C::Canvas, args...) = getindex(C.m, args...)

### General indexing support
function setindex!(C::Canvas, V, args...)
	setindex!(C.m, V, args...) # update CPU
	mark_for_update(C) # Note that this is the bottleneck for sequential single element updates; regardless, it should be quite fast.
end


################################
## Updating GPU mirror matrix ##
################################

# Uploads texture tex to GPU at address textureP[1]
# ToDo: should somehow figure out a way to check when this operation is done
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

function to_gpu(c::Canvas)
	textureP = [c.sprite.texture]
	map_to_rgb!(c)
	GLFW.MakeContextCurrent(c.window)
	to_gpu(c.rgb, textureP) 
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

# Linear indices support
# function setindex!(C::Canvas, val, ind) 
# 	i = CartesianIndices(C.m)[ind]
# 	setindex!(C, val, i)
# end
# function setindex!(C::Canvas, val, inds::Vector{CartesianIndex})
# 	if length(val) != length(inds); C.m[inds] = val; end # this will error appropriately
# 	for i in eachindex(1:length(inds))
# 		setindex!(C, val[i], inds[i])
# 	end
# end
# function setindex!(C::Canvas, val, ind::CartesianIndex)
# 	setindex!(C, val, ind.I[1], ind.I[2])
# end

# function setindex!(C::Canvas, val, i::Number, j::Number) # single value
# 	setindex!(C, [val], [i], [j])
# end

# Uploads texture tex to subarray (x,y) -> (x+w-1, y+w-1) of texture at address textureP[1] on GPU
# function to_gpu(tex::Array{UInt8,3}, textureP, x, y, w, h) 
#     size(tex)[1]!=3 && size(tex)[1]!=4 && (@error("texture format not supported"); return nothing)
#     isrgba = size(tex)[1]==4

# 	println("$x $y $w $h")

# 	glBindTexture(GL_TEXTURE_2D, textureP[1])
# 	if isrgba #rgba
# 		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, tex)
# 	else #rgb
# 		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_RGB, GL_UNSIGNED_BYTE, tex)
# 	end
# 	# glGenerateMipmap(GL_TEXTURE_2D)
# 	glFinish()
# end

# function to_gpu_sub(C::Canvas)
# 	GLFW.MakeContextCurrent(C.window)
# 	textureP = [C.sprite.texture]
# 	tex = map_to_rgb(C, C.up_minx:C.up_maxx, C.up_miny:C.up_maxy) # 1x1 array with the corresponding pixel in it
# 	x = C.up_minx -1
# 	y = C.up_miny -1
# 	w = C.up_maxx - C.up_minx + 1
# 	h = C.up_maxy - C.up_miny + 1
# 	to_gpu(tex, textureP, x, y, w, h) 
# end

### Experimental detailed indexing support
# function setindex!(C::Canvas, val, ind)
# 	@error("Linear indexing operation setindex!(::Canvas, $val, $inds) is not yet supported.")
# end

# setindex!(C::Canvas, V, ::Colon, Y) = setindex!(C,V,1:size(C.m)[1],Y)
# setindex!(C::Canvas, V, X, ::Colon) = setindex!(C,V,X,1:size(C.m)[2])
# setindex!(C::Canvas, V, ::Colon, ::Colon) = setindex!(C,V,1:size(C.m)[1],1:size(C.m)[2])

# function setindex!(C::Canvas, V, X, Y) # set V as submatrix of C at location X × Y
# 	# println("setindex! called with args $V, $X, $Y")

# 	if X === Colon(); X = 1:size(C.m)[1]; end
# 	if Y === Colon(); Y = 1:size(C.m)[2]; end

# 	minx, maxx = extrema(X)
# 	miny, maxy = extrema(Y)

# 	C.up_minx = min(C.up_minx, minx)
# 	C.up_maxx = max(C.up_maxx, maxx)
# 	C.up_miny = min(C.up_miny, miny)
# 	C.up_maxy = max(C.up_maxy, maxy)

# 	setindex!(C.m, V, X, Y)		
	
# 	t = time_ns()
# 	if t > C.next_update
# 		update!(C)
# 	else
# 		C.update_pending = true
# 	end
# end

# function update_pixel(C::Canvas, xp::Int, yp::Int)
# 	C.update_pending = false
# 	C.last_update = time_ns()
# 	# println("update pixel called")
# 	to_gpu(C, xp, yp) # should only happen on write to c.m
# end

end
