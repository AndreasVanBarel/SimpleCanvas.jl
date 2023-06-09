module SimpleCanvas

export canvas, close, setcolormap

# Exported temporarily for development purposes
export to_gpu, Canvas

using GLFW
using ModernGL
using LinearAlgebra

import Base: close
import Base: show
import Base: size, length, axes, getindex, setindex!

include("Sprite.jl")
include("GLPrograms.jl")
import .GLPrograms

programs = nothing

# Initializes GLFW library
function init()
	GLFW.Init() || @error("GLFW failed to initialize")

	# Specify OpenGL version
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 4);
	GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3);
	GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE);
    # GLFW.WindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); # ? Supposedly needed for MacOSX
end
init()

mutable struct Canvas{T} <: AbstractMatrix{T}
    const m::AbstractMatrix{T}
	colormap::Function # T -> UInt8[3] 
    window
	sprite
    last_update::UInt64 # last time that m was copied to canvas
	update_pending::Bool # whether there is a change to m that is not reflected on the canvas
	fps::Number
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

canvas(m::AbstractMatrix{T}, rgb::Function=rgb) where T = canvas(m, size(m)[2], size(m)[1], rgb)
function canvas(m::AbstractMatrix{T}, width::Integer, height::Integer, rgb::Function=rgb) where T 
    C = Canvas{T}(m, colormap_grayscale, nothing, nothing, 0, true, 10, 1, size(m)[1], 1, size(m)[2])
    C.window = createwindow(width, height, "Simple Canvas for matrix at $(UInt(pointer(m)))") # Create window
	GLFW.SetFramebufferSizeCallback(C.window, make_framebuffer_size_callback(C))
	tex = rgb(C, m)
	C.sprite = Sprite(tex) # Create sprite to show in window 
	update(C)
	task = @task polling_task(C)
	schedule(task)
    return C
end

function fill_update_zone(C::Canvas)
	C.update_pending = true
	C.up_minx = C.up_miny = 1
	C.up_maxx, C.up_maxy = size(C.m)
end 

function polling_task(C::Canvas)
	while !GLFW.WindowShouldClose(C.window)
		GLFW.MakeContextCurrent(C.window)
		# println("Polling task executing...")
		isnothing(C.window) && return
		if C.update_pending == true && time_ns()-C.last_update > 1e9/C.fps
			update(C) #TODO: This somewhat asynchronous update task might cause issues
		end
		GLFW.PollEvents()
		err = glGetError()
		err == 0 || @error("GL Error code $err")
		# C.last_update = time_ns()
		sleep(1/C.fps)
	end
	close(C)
end

# Create a window
function createwindow(width::Int=640, height::Int=480, title::String="SimpleCanvas window (badly initialized)")
	# Create a window and its OpenGL context
	window = GLFW.CreateWindow(width, height, title)
	isnothing(window) && @error("Window or context creation failed.")

	GLFW.MakeContextCurrent(window) # Make the window's context current
	glViewport(0, 0, width, height)

	GLFW.SwapInterval(1) #Activate V-sync

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
    println(io, "Canvas mirroring the matrix")
    show(io, C.m)
end
function show(io::IO, mime::MIME"text/plain", C::Canvas)
    println(io, "Canvas mirroring the matrix")
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
		#println("The window was resized to $width × $height")
		return nothing
	end
	return framebuffer_size_callback
end

##################
## Colormapping ##
##################

# Broadcast colormap on matrix
function rgb(C::Canvas, m::AbstractMatrix{T}) where T
	rgbs = Array{UInt8, 3}(undef, 3, size(m)[1], size(m)[2])
	# println("Colormap is $(C.colormap)")
	for c in CartesianIndices(m); rgbs[:,c] .= C.colormap(m[c]); end
	# println("rgbs = $(rgbs[3, 1:5, 1:5])")
	return rgbs
end

# A default colormap function
function colormap_grayscale(v::Float64) 
	v < 0 && (v = 0) # truncation
	v > 1 && (v = 1) # truncation
	fill(round(UInt8, v*255),3)
end
# Need function to convert from type T to pixel color values.
# Need function to rotate canvas (just changes quad vertices of course)

function setcolormap(C::Canvas, colormap::Function)
	C.colormap = colormap
	fill_update_zone(C)
	update(C)
	return
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
	# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) #GL_LINEAR
	# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); #GL_LINEAR_MIPMAP_LINEAR requires mipmaps
	#glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
	#glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
	#glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [0f0, 0f0, 0f0, 0f0])
	# glFinish() # supposedly blocks until GL upload finished
end

function to_gpu(c::Canvas)
	# println("to_gpu called")
	GLFW.MakeContextCurrent(c.window)
	textureP = [c.sprite.texture]
	tex = rgb(c, c.m)
	to_gpu(tex, textureP) 
end

# Uploads texture tex to subarray (x,y) -> (x+w-1, y+w-1) of texture at address textureP[1] on GPU
function to_gpu(tex::Array{UInt8,3}, textureP, x, y, w, h) 
    size(tex)[1]!=3 && size(tex)[1]!=4 && (@error("texture format not supported"); return nothing)
    isrgba = size(tex)[1]==4

	println("$x $y $w $h")

	glBindTexture(GL_TEXTURE_2D, textureP[1])
	if isrgba #rgba
		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, tex)
	else #rgb
		glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_RGB, GL_UNSIGNED_BYTE, tex)
	end
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); #GL_LINEAR_MIPMAP_LINEAR requires mipmaps
	# glGenerateMipmap(GL_TEXTURE_2D)
	glFinish()
end

function to_gpu_sub(c::Canvas)
	GLFW.MakeContextCurrent(c.window)
	textureP = [c.sprite.texture]
	tex = rgb(c, c.m[c.up_minx:c.up_maxx, c.up_miny:c.up_maxy]) # 1x1 array with the corresponding pixel in it
	x = c.up_minx -1
	y = c.up_miny -1
	w = c.up_maxx - c.up_minx + 1
	h = c.up_maxy - c.up_miny + 1
	to_gpu(tex, textureP, x, y, w, h) 
end

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

function setindex!(C::Canvas, val, ind)
	@error("Linear indexing operation setindex!(::Canvas, $val, $inds) is not yet supported.")
end

setindex!(C::Canvas, V, ::Colon, Y) = setindex!(C,V,1:size(C.m)[1],Y)
setindex!(C::Canvas, V, X, ::Colon) = setindex!(C,V,X,1:size(C.m)[2])
setindex!(C::Canvas, V, ::Colon, ::Colon) = setindex!(C,V,1:size(C.m)[1],1:size(C.m)[2])

function setindex!(C::Canvas, V, X, Y) # set V as submatrix of C at location X × Y
	# println("setindex! called with args $V, $X, $Y")

	if X === Colon(); X = 1:size(C.m)[1]; end
	if Y === Colon(); Y = 1:size(C.m)[2]; end

	minx, maxx = extrema(X)
	miny, maxy = extrema(Y)

	C.up_minx = min(C.up_minx, minx)
	C.up_maxx = max(C.up_maxx, maxx)
	C.up_miny = min(C.up_miny, miny)
	C.up_maxy = max(C.up_maxy, maxy)

	setindex!(C.m, V, X, Y)		
	
	t = time_ns()
	if t-C.last_update > 1e9/C.fps #longer than 0.1 sec ago update
		update(C)
	else
		C.update_pending = true
	end
end
function update(C::Canvas)
	isnothing(C.window) && return
	GLFW.MakeContextCurrent(C.window)
	# println("Update zone [$(C.up_minx):$(C.up_maxx)] × [$(C.up_miny):$(C.up_maxy)]")
	# println("C.colormap is $(C.colormap)")

	to_gpu(C) # pushes data to be updated to the GPU
	redraw(C) # instructs the GPU to redraw

	# reset updating area to none
	C.update_pending = false
	C.up_minx, C.up_miny = size(C.m)
	C.up_maxx = C.up_maxy = 1

	C.last_update = time_ns()
end
# function update_pixel(C::Canvas, xp::Int, yp::Int)
# 	C.update_pending = false
# 	C.last_update = time_ns()
# 	# println("update pixel called")
# 	to_gpu(C, xp, yp) # should only happen on write to c.m
# end

# does not push anything to gpu, just redraws
function redraw(C::Canvas)
	draw(C.sprite)
	GLFW.SwapBuffers(C.window)
end

end
