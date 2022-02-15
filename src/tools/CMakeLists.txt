add_executable(labelstats labelstats.cpp)

target_link_libraries(labelstats PRIVATE CLI11::CLI11 libdismec)

add_executable(tfidf tfidf.cpp)

target_link_libraries(tfidf PRIVATE CLI11::CLI11 libdismec)
