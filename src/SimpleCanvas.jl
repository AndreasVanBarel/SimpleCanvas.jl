module SimpleCanvas

export canvas, close, colormap!, name!, diagnostic_level!, target_fps! #, show_fps!
export vsync!

# Exported temporarily for development purposes
export Canvas, to_gpu, map_to_rgb!

using GLFW
using ModernGL
using LinearAlgebra

# using Base.Threads

import Base: close
import Base: show
import Base: size, length, axes, getindex, setindex!

include("Sprite.jl")
include("GLPrograms.jl")
import .GLPrograms

# Settings
default_fps = 60
max_time_fraction = 0.25
default_diagnostic_level = 0
default_updates_immediately = true

programs = nothing

# Initializes GLFW library
function init()
	GLFW.Init() || @error("GLFW failed to initialize")

	# Specify OpenGL version
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3);
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3);
	GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE);
    # GLFW.WindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); # ? Supposedly needed for MacOSX
end
init()

mutable struct Canvas{T} <: AbstractMatrix{T}
	# Core functionality
    m::Matrix{T}
	rgb::Array{UInt8, 3} # RGB(A) matrix (preallocated for performance)
	colormap::Function # T -> UInt8[3] 
    window
	sprite

	# Timing and update management
    last_update::UInt64 # last time that update() was called (time at the start of the call)
	next_update::UInt64 # next time that update() could (and should) be called (set at the end of the previous update() call)
	update_pending::Bool # whether there is a change to m that is not uploaded to the GPU and thus not reflected on the canvas
	updates_immediately::Bool
	fps::Number

	# Additional functionality
	show_fps::Bool
	diagnostic_level::Int # 0: none, 1: uploads to GPU, 2: + redraws
	name::String
	up_minx::Int # These four determine that the submatrix to be updated is contained within
	up_maxx::Int # m[up_minx:up_maxx, up_miny:up_maxy]
	up_miny::Int 
	up_maxy::Int
    # finalizing (releasing memory when canvas becomes inaccessible)
    # function Canvas{T}(m::Matrix{T}, rgb::Function) where T
    #     C = new(m, rgb, nothing, nothing, 0)
    #     function f(C)
    #         @async println("Finalizing $C since there are no pointers to C anymore")
    #         close(C)
    #     end
    #     finalizer(f, C)
    # end
end

# TODO: Create a better construction interface, probably with Canvas type itself as name
# such that Canvas{T}(w,h) creates a canvas of size w×h
canvas(m::AbstractMatrix{T}) where T = canvas(m, size(m)[2], size(m)[1]; name = "Simple Canvas")
function canvas(m::AbstractMatrix{T}, width::Integer, height::Integer; name::String) where T 
	# println("Creating canvas for matrix at $(UInt(pointer(m)))")
	m = collect(m)::Matrix{T}
	rgb = Array{UInt8, 3}(undef, 3, size(m)[1], size(m)[2])
    C = Canvas{T}(m, rgb, colormap_grayscale, nothing, nothing, 
		0, 0, true, default_updates_immediately, default_fps, 
		true, default_diagnostic_level, name, 
		1, size(m)[1], 1, size(m)[2])

	# Create OpenGL context
    C.window = createwindow(width, height, name) # Create window
	GLFW.SetFramebufferSizeCallback(C.window, make_framebuffer_size_callback(C))
	map_to_rgb!(C)
	tex = C.rgb
	C.sprite = Sprite(tex) # Create sprite to show in window 

	# Create the polling task
	task = @task polling_task(C)
	schedule(task)
	# task = Threads.@spawn polling_task(C) 
    return C
end

function polling_task(C::Canvas)
	while !GLFW.WindowShouldClose(C.window)
		t_start = time_ns()
		# did_update = false
		GLFW.MakeContextCurrent(C.window)
		# println("Polling task executing... $(C.update_pending), $(time_ns()), $(C.next_update)")
		isnothing(C.window) && return
		if C.update_pending == true && time_ns() > C.next_update
			did_update = true
			C.diagnostic_level >= 1 && println("Canvas '$(C.name)': upload to GPU & redraw (initiated by polling task)")
			update!(C) 
		else 
			C.diagnostic_level >= 2 && println("Canvas '$(C.name)': redraw (initiated by polling task)")
			redraw(C) # Redraws the canvas if no update is pending; this ensures the window is responsive
			# if C.show_fps
			# 	last_redraw = max(last_redraw_only, C.last_update)
			# 	fps = round(Int,1e9/(time_ns()-last_redraw))
			# 	GLFW.SetWindowTitle(C.window, "$(C.name) @ $(fps)fps ($(C.fps))")
			# end
		end
		GLFW.PollEvents()
		err = glGetError()
		err == 0 || @error("GL Error code $err")
		# if did_update
		# 	Δt = time_ns() - t_start
		# 	println("Δt = $(1e-9Δt)s ($(round(Int,1e9/Δt)) fps)")
		# 	Δt_next = C.next_update - time_ns()
		# 	# println("C.next_update: $(C.next_update), which is in $(1e-9Δt_next)s ($(round(Int,1e9/Δt_next)) fps")
		# end
		sleep(1/C.fps)
	end
	close(C)
end

function createwindow(width::Int=640, height::Int=480, title::String="SimpleCanvas window")
	# Create a window and its OpenGL context
	window = GLFW.CreateWindow(width, height, title)
	isnothing(window) && @error("Window or context creation failed.")

	GLFW.MakeContextCurrent(window) # Make the window's context current
	glViewport(0, 0, width, height)

	GLFW.SwapInterval(0) #Activate V-sync (not necessary since we are using sleep in the polling task)

	# Callback functions
	# GLFW.SetErrorCallback(error_callback)
	# GLFW.SetKeyCallback(window, key_callback)
	# GLFW.SetMouseButtonCallback(window, mouse_button_callback)

    # Compile shader programs
	global programs = GLPrograms.generatePrograms()
	glUseProgram(programs.sprite); glUniformMatrix3fv(programs.sprite_camLoc, 1, false, Matrix{Float32}(I,3,3)); # Set Camera values on the gpu

	gpu_allocate() # Allocate buffers on the GPU	

	return window
end

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
	isnothing(C.window) && return
    GLFW.SetWindowShouldClose(C.window, true)
    GLFW.DestroyWindow(C.window)
	C.window = nothing
    return
end

function show(io::IO, C::Canvas) 
    println(io, "$(typeof(C)) named '$(C.name)'")
    show(io, C.m)
end
function show(io::IO, mime::MIME"text/plain", C::Canvas)
    println(io, "$(typeof(C)) named '$(C.name)'")
    show(io, mime, C.m)
end

function reset()
    GLFW.Terminate()
    init()
end

function make_framebuffer_size_callback(C)
	function framebuffer_size_callback(window::GLFW.Window, width, height)
		glViewport(0, 0, width, height);
		redraw(C)
		# println("The window was resized to $width × $height")
		return nothing
	end
	return framebuffer_size_callback
end

# Setting options 
function name!(C::Canvas, name::String)
	C.name = name
	GLFW.SetWindowTitle(C.window, name);
end

function diagnostic_level!(C::Canvas, level::Int)
	C.diagnostic_level = level
end

function target_fps!(C::Canvas, fps::Number)
	C.fps = fps
end

function show_fps!(C::Canvas)
	C.show_fps = true
end

function vsync!(C::Canvas, vsync::Bool)
	GLFW.MakeContextCurrent(C.window)
	GLFW.SwapInterval(vsync ? 1 : 0)
end

# NOTE: Unused function
# function fill_update_zone(C::Canvas)
# 	C.update_pending = true
# 	C.up_minx = C.up_miny = 1
# 	C.up_maxx, C.up_maxy = size(C.m)
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
	C.update_pending = true
	if C.updates_immediately
		t = time_ns()
		if t > C.next_update
			C.diagnostic_level >= 1 && println("Canvas '$(C.name)': upload to GPU & redraw (initiated by setcolormap)")
			update!(C)
		end
	end
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
	GLFW.MakeContextCurrent(c.window)
	textureP = [c.sprite.texture]
	map_to_rgb!(c)
	to_gpu(c.rgb, textureP) 
end

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

# function to_gpu_sub(c::Canvas)
# 	GLFW.MakeContextCurrent(c.window)
# 	textureP = [c.sprite.texture]
# 	tex = map_to_rgb(c, c.up_minx:c.up_maxx, c.up_miny:c.up_maxy) # 1x1 array with the corresponding pixel in it
# 	x = c.up_minx -1
# 	y = c.up_miny -1
# 	w = c.up_maxx - c.up_minx + 1
# 	h = c.up_maxy - c.up_miny + 1
# 	to_gpu(tex, textureP, x, y, w, h) 
# end

###################################
## Matrix operations on a Canvas ##
###################################

size(C::Canvas)	= size(C.m)
length(C::Canvas) = length(C.m)	
axes(C::Canvas) = axes(C.m)	
getindex(C::Canvas, args...) = getindex(C.m, args...)

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

### General indexing support
function setindex!(C::Canvas, V, args...)
	setindex!(C.m, V, args...) # update CPU

	C.update_pending = true
	if C.updates_immediately
		t = time_ns()
		if t > C.next_update
			C.diagnostic_level >= 1 && println("Canvas '$(C.name)': upload to GPU & redraw (initiated by setindex!)")
			update!(C)
		end
	end
end

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
# 		update(C)
# 	else
# 		C.update_pending = true
# 	end
# end

function update!(C::Canvas)
	# println("Update zone [$(C.up_minx):$(C.up_maxx)] × [$(C.up_miny):$(C.up_maxy)]")

	C.last_update = time_ns()
	C.update_pending = false

	isnothing(C.window) && return
	GLFW.MakeContextCurrent(C.window)

	# reset updating area to none
	C.up_minx, C.up_miny = size(C.m)
	C.up_maxx = C.up_maxy = 1

	to_gpu(C) # pushes data to be updated to the GPU
	redraw(C) # instructs the GPU to redraw

	# In case the update took a long time, we need to provide the caller with some additional time to do his business. We make it so that at most 50pct of the time goes to the canvas update. If the update takes longer than half the frame time, we allow the caller at least that same amount of time. The target fps will then not be reached.
	t = time_ns()
	update_time_elapsed = t - C.last_update
	if update_time_elapsed > 1e9/C.fps*max_time_fraction
		C.next_update = C.last_update + round(UInt64, update_time_elapsed/max_time_fraction)
	else 
		C.next_update = C.last_update + round(UInt64, 1e9/C.fps)
	end
	if C.diagnostic_level >= 1
		Δupdate = C.next_update - C.last_update
		println("Canvas '$(C.name)': updated in $(1e-9update_time_elapsed)s, next in $(1e-9update_time_elapsed)s ($(round(Int,1e9/Δupdate)) fps)")
	end
end
# function update_pixel(C::Canvas, xp::Int, yp::Int)
# 	C.update_pending = false
# 	C.last_update = time_ns()
# 	# println("update pixel called")
# 	to_gpu(C, xp, yp) # should only happen on write to c.m
# end

# does not push anything to gpu, just redraws. Is extremely fast (often <10μs)
function redraw(C::Canvas)
	# time_start = time_ns()
	draw(C.sprite)
	GLFW.SwapBuffers(C.window)
	# Δt = time_ns() - time_start
	# println("Redraw took $(1e-9Δt)s ($(round(Int,1e9/Δt)) fps)")
end

end
