add_library( gslam SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry.cpp
    Optimizer.cpp
)

target_link_libraries( gslam
    ${THIRD_PARTY_LIBS}
)
