CXX=g++
CXXFLAGS=-std=c++17 -Wall -I../include

TARGET=fp8.test

$(TARGET):main.cpp ../include/simple_fp8.hpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
