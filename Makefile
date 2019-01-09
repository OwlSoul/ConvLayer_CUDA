CXX = nvcc
TARGET = cnnConvLayer

$(info )
$(info          CUDA HATING SHARK        )
$(info                                   )
$(info _________         .    .          )
$(info ..       \_    ,  |\  /|         )
$(info \       0  \  /|  \ \/ /          )
$(info ~\______    \/ |   \  /           )
$(info ~~~vvvv\    \ |   /  |            )
$(info ~~~~\^^^^  ==   \_/   |            )
$(info ~~~~`\_   ===    \.  |            )
$(info ~~~~~/ /\_   \ /      |            )
$(info ~~~~~|/   \_  \|     /            )
$(info ~~~~~~~~~~~\________/ I HATE CUDA!)
$(info )
$(info Wait until the program compiles. Be patient. Thank you.)
$(info Sincerely, Team 8.)
$(info )
$(info )


all: cnnConvLayer.cu
	$(CXX) $< -o $(TARGET) -arch compute_30 -code sm_30 -use_fast_math

.PHONY: clean run

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)
