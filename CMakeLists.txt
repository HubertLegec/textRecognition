cmake_minimum_required(VERSION 3.7)
project(textRecognition)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/TextDetector.cpp src/TextDetector.h src/TextRecognizer.cpp src/TextRecognizer.h src/ImageLoader.cpp src/ImageLoader.h )
add_executable(textRecognition ${SOURCE_FILES})
target_link_libraries(textRecognition ${OpenCV_LIBS} )