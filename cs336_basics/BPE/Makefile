# 1. 自动获取 Python 和 pybind11 的头文件包含路径
PYTHON_INCLUDES := $(shell python3 -m pybind11 --includes)

# 2. 编译目标和参数
TARGET = merges_algo$(shell python3-config --extension-suffix)
SRC = merges.cpp
CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC $(PYTHON_INCLUDES)

# 默认规则：编译并自动配置 VS Code
all: $(TARGET) vscode_config

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# 核心黑科技：运行 make 时自动刷新 VS Code 的包含路径，消灭红线！
vscode_config:
	@mkdir -p .vscode
	@echo "{" > .vscode/c_cpp_properties.json
	@echo "    \"configurations\": [" >> .vscode/c_cpp_properties.json
	@echo "        {" >> .vscode/c_cpp_properties.json
	@echo "            \"name\": \"Auto-Configured\"," >> .vscode/c_cpp_properties.json
	@echo "            \"includePath\": [" >> .vscode/c_cpp_properties.json
	@echo "                \"\$${workspaceFolder}/**\"," >> .vscode/c_cpp_properties.json
	@for path in $(shell python3 -m pybind11 --includes | sed 's/-I//g'); do \
		echo "                \"$$path\"," >> .vscode/c_cpp_properties.json; \
	done
	@echo "                \"/usr/include\"" >> .vscode/c_cpp_properties.json
	@echo "            ]" >> .vscode/c_cpp_properties.json
	@echo "        }" >> .vscode/c_cpp_properties.json
	@echo "    ]," >> .vscode/c_cpp_properties.json
	@echo "    \"version\": 4" >> .vscode/c_cpp_properties.json
	@echo "}" >> .vscode/c_cpp_properties.json
	@echo "[Makefile] VS Code 头文件路径已自动配置，红线已消除！"

clean:
	rm -f *.so