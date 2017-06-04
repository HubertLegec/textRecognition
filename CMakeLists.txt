cmake_minimum_required(VERSION 3.7)
project(textRecognition)

find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 11)

set(JAVA_INCLUDE /Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home/include /Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home/include/darwin)

set(SOURCE_FILES src/TextDetector.cpp src/TextDetector.h src/TextRecognizer.cpp src/TextRecognizer.h src/ImageLoader.cpp src/ImageLoader.h src/DetectedWord.cpp src/DetectedWord.h src/RecognitionModule.cpp src/RecognitionModule.h)
add_executable(textRecognition ${SOURCE_FILES}  src/main.cpp)
add_library(recognitionModule ${SOURCE_FILES} wrap/recognition_module_wrap.cpp)
add_custom_command(TARGET recognitionModule
        PRE_BUILD
        COMMAND mkdir -p com/legec/recognitionmodule
        COMMAND mkdir -p ../wrap
        COMMAND swig -java -c++ -package com.legec.recognitionmodule -outdir com/legec/recognitionmodule -o ../wrap/recognition_module_wrap.cpp ../src/recognitionModule.i)

target_link_libraries(textRecognition SHARED ${OpenCV_LIBS})
target_link_libraries(recognitionModule ${JAVA_INCLUDE})
target_link_libraries(recognitionModule ${OpenCV_LIBS})