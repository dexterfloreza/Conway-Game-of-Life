cmake_minimum_required(VERSION 3.16)
project(SFML_GameOfLife)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Force Release build" FORCE)

# Path to SFML 2.6.1
set(SFML_DIR "C:/Users/DeRya/Downloads/SFML-2.6.1-windows-gcc-13.1.0-mingw-64-bit/SFML-2.6.1/lib/cmake/SFML")

find_package(SFML 2.6 COMPONENTS system window graphics audio REQUIRED)

# --- Main executable target ---
add_executable(SFML_GameOfLife
        main.cpp
)

target_link_libraries(SFML_GameOfLife
        sfml-system
        sfml-window
        sfml-graphics
        sfml-audio
)

# --- Automatically copy DLLs to the build folder ---
add_custom_command(TARGET SFML_GameOfLife POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/Users/DeRya/Downloads/SFML-2.6.1-windows-gcc-13.1.0-mingw-64-bit/SFML-2.6.1/bin"
        $<TARGET_FILE_DIR:SFML_GameOfLife>
)
