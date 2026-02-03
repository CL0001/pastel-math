include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

function(add_pastelmath_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_link_libraries(
        ${TEST_NAME} 
        PRIVATE 
        PastelMath::PastelMath
        GTest::gtest_main
    )
    
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    set_tests_properties(
        ${TEST_NAME} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

add_pastelmath_test(VecTest)
add_pastelmath_test(MatTest)
add_pastelmath_test(QuatTest)