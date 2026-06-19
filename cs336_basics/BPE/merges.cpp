#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 🌟 核心：有了它，C++ 的 std::pair 会自动映射为 Python 的 tuple
#include <iostream>
#include <stdio.h>
#include <string>
#include <set>
#include <algorithm>
#include <map>
#include <vector>

namespace py = pybind11;

// 1. 排行榜节点数据结构
class byte_pairs {
public:
    std::string bytes1;
    std::string bytes2;
    int nums;

    byte_pairs(int nums, std::string b1, std::string b2) 
        : bytes1(std::move(b1)), bytes2(std::move(b2)), nums(nums) {}

    bool operator < (const byte_pairs& other) const {
        if (this->nums != other.nums) {
            return this->nums > other.nums; // 高频在前
        }
        if (this->bytes1 != other.bytes1) {
            return this->bytes1 > other.bytes1; 
        }
        return this->bytes2 > other.bytes2;
    }
};

// 2. 轻量化倒排索引链表
class string_node {
public:
    string_node* next;
    size_t pre_tokens_index; 
    size_t pos;              
    
    string_node(size_t ide) : next(nullptr), pre_tokens_index(ide), pos(0) {}
    string_node(size_t ide, size_t pos_) : next(nullptr), pre_tokens_index(ide), pos(pos_) {}
    string_node(size_t ide, size_t pos_, string_node* next_) : next(next_), pre_tokens_index(ide), pos(pos_) {}
    string_node() : next(nullptr), pre_tokens_index(0), pos(0) {}
};

class string_list_hander{
    public:
    string_node* first;
    string_node* last;
    void insert(size_t ide, size_t pos){
        assert(last != nullptr);
        string_node *node = new string_node(ide);
        node->pos = pos;
        node->next = nullptr;
        last->next = node;
        last = node;
    }
    ~string_list_hander(){
        string_node* cur = this->first;
        while(cur!=nullptr){
            string_node* next = cur->next;
            delete cur; cur = next;
        }
    }
};

static string_list_hander* create_string_list_hander(size_t index, size_t pos) {
    string_list_hander* handler = new string_list_hander;
    handler->first = handler->last = new string_node(index, pos, nullptr);
    return handler;
}

// 3. 【完全体训练函数】
// 返回值：std::pair< 词表, Merges历史表 >
std::pair<std::vector<py::bytes>, std::vector<std::pair<py::bytes, py::bytes>>> 
run_cpp_algorithm(const py::dict& py_dict, int merge_times) {
    
    std::set<byte_pairs> algorithm_set;
    std::map<std::pair<std::string, std::string>, int> counter_map;
    std::map<std::pair<std::string, std::string>, string_list_hander *> pair_to_word_map;
    
    // 返回容器 A：训练好的新 Token 词表
    std::vector<py::bytes> trained_vocab; 
    // 🌟 返回容器 B：每次合并的详细信息表 list[tuple[bytes, bytes]]
    std::vector<std::pair<py::bytes, py::bytes>> merges_history; 

    std::vector<std::string> all_pre_tokens;
    std::vector<std::set<int>> merge_edge;
    std::vector<int> word_weights; 

    
    size_t index = 0;
    for (auto item : py_dict){
        std::string raw_key = item.first.cast<py::bytes>();
        int value_nums = item.second.cast<int>();
        int len_raw_key = raw_key.length();
        if (len_raw_key <= 1) continue;

        all_pre_tokens.push_back(raw_key);
        merge_edge.push_back(std::set<int>()); 
        word_weights.push_back(value_nums); 

        for(int i = 0; i < len_raw_key - 1; i++){
            auto it = std::make_pair(raw_key.substr(i, 1), raw_key.substr(i + 1, 1));
            counter_map[it] += value_nums;
            if(pair_to_word_map.find(it) == pair_to_word_map.end()){
                pair_to_word_map[it] = create_string_list_hander(index, i);
                merge_edge[index].insert(i + 1);
                continue;
            }
            pair_to_word_map[it]->insert(index, i);
            merge_edge[index].insert(i + 1);
        }
        index++;
    }

    for (auto item : counter_map) {
        algorithm_set.insert(byte_pairs(item.second, item.first.first, item.first.second));
    }
    
    for (int i = 0; i < merge_times; i++){
        if (algorithm_set.empty()) break;

        byte_pairs max_fre_pair = *algorithm_set.begin();
        algorithm_set.erase(algorithm_set.begin());

        auto current_pair_key = std::make_pair(max_fre_pair.bytes1, max_fre_pair.bytes2);
        std::string new_token = max_fre_pair.bytes1 + max_fre_pair.bytes2;

        if (max_fre_pair.nums != counter_map[current_pair_key]) {
            i--; continue;
        }
        if (counter_map[current_pair_key] <= 0) continue;

        // 记录新产生的 Token 到词表
        trained_vocab.push_back(py::bytes(new_token));
        
        // 🌟 核心新增：把本次成功合并的左 bytes 和右 bytes 整成 pair 并打入历史表
        merges_history.push_back(std::make_pair(py::bytes(max_fre_pair.bytes1), py::bytes(max_fre_pair.bytes2)));

        auto head_ptr = pair_to_word_map[current_pair_key];
        if (head_ptr == nullptr) continue;
        // if (i%100 == 0)printf("index:%d\n",i);
        for (auto i_node = head_ptr->first; i_node != nullptr; i_node = i_node->next) {
            size_t pre_tokens_index = i_node->pre_tokens_index;
            size_t pos = i_node->pos;
            
            std::string& raw_word = all_pre_tokens[pre_tokens_index];
            std::set<int>& edges = merge_edge[pre_tokens_index];
            int w_count = word_weights[pre_tokens_index];

            size_t wall_pos = pos + max_fre_pair.bytes1.length();
            auto curr_wall_it = edges.find(wall_pos);
            if (curr_wall_it == edges.end()) {
                continue; 
            }

            size_t left_bound = (curr_wall_it == edges.begin()) ? 0 : *std::prev(curr_wall_it);
            auto next_wall_it = std::next(curr_wall_it);
            size_t right_bound = (next_wall_it == edges.end()) ? raw_word.length() : *next_wall_it;

            std::string current_left_token  = raw_word.substr(left_bound, wall_pos - left_bound);
            std::string current_right_token = raw_word.substr(wall_pos, right_bound - wall_pos);

            if (current_left_token == max_fre_pair.bytes1 && current_right_token == max_fre_pair.bytes2) {
                
                // 【步骤 1】：扣减老邻居
                if (left_bound > 0) {
                    auto left_bound_it = std::prev(curr_wall_it);
                    size_t leftleft_bound = (left_bound_it == edges.begin()) ? 0 : *std::prev(left_bound_it);
                    std::string left_neighbor = raw_word.substr(leftleft_bound, left_bound - leftleft_bound);
                    
                    auto old_left_pair = std::make_pair(left_neighbor, current_left_token);
                    counter_map[old_left_pair] -= w_count;
                    algorithm_set.insert(byte_pairs(counter_map[old_left_pair], old_left_pair.first, old_left_pair.second));
                }
                if (right_bound < raw_word.length()) {
                    auto right_right_it = std::next(next_wall_it);
                    size_t right_right_bound = (right_right_it == edges.end()) ? raw_word.length() : *right_right_it;
                    std::string right_neighbor = raw_word.substr(right_bound, right_right_bound - right_bound);
                    
                    auto old_right_pair = std::make_pair(current_right_token, right_neighbor);
                    counter_map[old_right_pair] -= w_count;
                    algorithm_set.insert(byte_pairs(counter_map[old_right_pair], old_right_pair.first, old_right_pair.second));
                }

                // 【步骤 2】：擦除边界墙
                edges.erase(curr_wall_it); 

                // 【步骤 3】：累加新邻居并动态反向回注新火种
                if (left_bound > 0) {
                    auto prev_it = edges.find(left_bound);
                    size_t leftleft_bound = (prev_it == edges.begin()) ? 0 : *std::prev(prev_it);
                    std::string left_neighbor = raw_word.substr(leftleft_bound, left_bound - leftleft_bound);
                    auto new_left_pair = std::make_pair(left_neighbor, new_token);
                    counter_map[new_left_pair] += w_count;
                    algorithm_set.insert(byte_pairs(counter_map[new_left_pair], new_left_pair.first, new_left_pair.second));
                    
                    if (pair_to_word_map.find(new_left_pair) == pair_to_word_map.end()) {
                        pair_to_word_map[new_left_pair] = create_string_list_hander(pre_tokens_index, leftleft_bound);
                    }
                    else{
                        pair_to_word_map[new_left_pair]->insert(pre_tokens_index, leftleft_bound);
                    }
                }

                if (right_bound < raw_word.length()) {
                    auto right_curr_it = edges.find(right_bound);
                    auto right_next_it = std::next(right_curr_it);
                    size_t right_right_bound = (right_next_it == edges.end()) ? raw_word.length() : *right_next_it;
                    std::string right_neighbor = raw_word.substr(right_bound, right_right_bound - right_bound);
                    
                    auto new_right_pair = std::make_pair(new_token, right_neighbor);
                    counter_map[new_right_pair] += w_count;
                    algorithm_set.insert(byte_pairs(counter_map[new_right_pair], new_right_pair.first, new_right_pair.second));
                    
                    if (pair_to_word_map.find(new_right_pair) == pair_to_word_map.end()) {
                        pair_to_word_map[new_right_pair] = create_string_list_hander(pre_tokens_index, left_bound);
                    }
                    else{
                        pair_to_word_map[new_right_pair]->insert(pre_tokens_index, left_bound);
                    }
                }
            }   
        }
        counter_map[current_pair_key] = 0;
    }

    // 清理倒排链表堆内存
    for (auto item : pair_to_word_map) {
        delete item.second;
    }
    
    // 🌟 返回包含了词表和 Merges 历史表的巨大 Pair 组合
    return std::make_pair(trained_vocab, merges_history); 
}

PYBIND11_MODULE(merges_algo, m) {
    m.def("run_algorithm", &run_cpp_algorithm, "Runs bytes BPE trainer and returns both vocab and merges history",
          py::arg("py_dict"), py::arg("merge_times"));
}