module SimplerCanvas

export spawn_window

using GLFW
using Base.Threads

function create_window()
    println("creating window on thread $(Threads.threadid())")
    GLFW.Init()  # Initialize GLFW

    window = GLFW.CreateWindow(800, 600, "GLFW Window")

    if window == C_NULL
        error("Failed to create GLFW window")
    end

    GLFW.MakeContextCurrent(window)

    while !GLFW.WindowShouldClose(window)
        GLFW.PollEvents()  # Process events
        GLFW.SwapBuffers(window)  # Swap front and back buffers
    end

    GLFW.DestroyWindow(window)
    GLFW.Terminate()
end

spawn_window() = Threads.@spawn create_window()

end