set(CMAKE_INCLUDE_CURRENT_DIR on)

include_directories(${INCLUDE_DIR} ${ODINDATA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${LOG4CXX_INCLUDE_DIRS}/.. ${ZEROMQ_INCLUDE_DIRS})

file(GLOB APP_SOURCES XspressController.cpp XspressDetector.cpp XspressDAQ.cpp ILibXspress.cpp LibXspressWrapper.cpp LibXspressSimulator.cpp)

add_executable(xspressControl ${APP_SOURCES} XspressControlApp.cpp)

target_link_libraries(
    xspressControl
    ${XSPRESS_LIBRARIES}
    ${ODINDATA_LIBRARIES}
    ${Boost_LIBRARIES}
    ${LOG4CXX_LIBRARIES}
    ${ZEROMQ_LIBRARIES}
    ${LIBXSPRESS_LIBRARIES}
)

if ( ${CMAKE_SYSTEM_NAME} MATCHES Linux )
    find_library(PTHREAD_LIBRARY
             NAMES pthread)
    target_link_libraries(xspressControl ${PTHREAD_LIBRARY} )
endif()

install(TARGETS xspressControl RUNTIME DESTINATION bin)

