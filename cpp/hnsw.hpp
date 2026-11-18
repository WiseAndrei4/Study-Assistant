#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

class HNSWIndex {
public:
    HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200, uint32_t seed = 42)
        : dim_(dim), M_(M), M_max0_(M * 2), ef_construction_(ef_construction),
          m_L_(1.0 / std::log(static_cast<double>(M))), rng_(seed) {}

    void add_point(const float* vec, uint32_t id) {
        size_t idx = vectors_.size() / dim_;
        vectors_.insert(vectors_.end(), vec, vec + dim_);
        ids_.push_back(id);

        int level = sample_level();
        if (static_cast<int>(links_.size()) <= level) {
            links_.resize(level + 1);
        }
        for (int l = 0; l <= level; ++l) {
            if (links_[l].size() <= idx) links_[l].resize(idx + 1);
        }

        if (idx == 0) {
            enter_point_ = 0;
            max_level_ = level;
            return;
        }

        uint32_t curr_obj = enter_point_;
        float curr_dist = distance(vec, get_vec(curr_obj));

        // 1. Greedy descent from max_level_ down to level + 1
        for (int l = max_level_; l > level; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t neighbor : links_[l][curr_obj]) {
                    float d = distance(vec, get_vec(neighbor));
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // 2. Beam search and bidirectional link updates from min(level, max_level_) down to 0
        for (int l = std::min(level, max_level_); l >= 0; --l) {
            auto top_candidates = search_layer(vec, curr_obj, ef_construction_, l);
            size_t max_links = (l == 0) ? M_max0_ : M_;
            auto neighbors = select_neighbors(top_candidates, max_links);

            links_[l][idx] = neighbors;
            for (uint32_t nb : neighbors) {
                auto& nb_links = links_[l][nb];
                nb_links.push_back(static_cast<uint32_t>(idx));
                if (nb_links.size() > max_links) {
                    std::vector<std::pair<float, uint32_t>> scored;
                    scored.reserve(nb_links.size());
                    const float* nb_vec = get_vec(nb);
                    for (uint32_t cand : nb_links) {
                        scored.emplace_back(distance(nb_vec, get_vec(cand)), cand);
                    }
                    std::sort(scored.begin(), scored.end());
                    nb_links.resize(max_links);
                    for (size_t k = 0; k < max_links; ++k) nb_links[k] = scored[k].second;
                }
            }
            if (!top_candidates.empty()) curr_obj = top_candidates.back().second;
        }

        if (level > max_level_) {
            max_level_ = level;
            enter_point_ = static_cast<uint32_t>(idx);
        }
    }

    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k, size_t ef_search) const {
        if (ids_.empty()) return {};
        ef_search = std::max(ef_search, k);

        uint32_t curr_obj = enter_point_;
        float curr_dist = distance(query, get_vec(curr_obj));

        for (int l = max_level_; l > 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t neighbor : links_[l][curr_obj]) {
                    float d = distance(query, get_vec(neighbor));
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        auto results = search_layer(query, curr_obj, ef_search, 0);
        if (results.size() > k) results.resize(k);

        std::vector<std::pair<uint32_t, float>> out;
        out.reserve(results.size());
        for (const auto& [dist, internal_idx] : results) {
            out.emplace_back(ids_[internal_idx], 1.0f - dist); // Cosine similarity
        }
        return out;
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open index file for writing");
        uint32_t magic = 0x484E5357; // "HNSW"
        uint64_t n = ids_.size();
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&enter_point_), sizeof(enter_point_));
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(ids_.data()), n * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(vectors_.data()), vectors_.size() * sizeof(float));

        uint64_t num_layers = links_.size();
        out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        for (const auto& layer : links_) {
            uint64_t layer_nodes = layer.size();
            out.write(reinterpret_cast<const char*>(&layer_nodes), sizeof(layer_nodes));
            for (const auto& adj : layer) {
                uint32_t deg = static_cast<uint32_t>(adj.size());
                out.write(reinterpret_cast<const char*>(&deg), sizeof(deg));
                out.write(reinterpret_cast<const char*>(adj.data()), deg * sizeof(uint32_t));
            }
        }
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open index file for reading");
        uint32_t magic = 0;
        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x484E5357) throw std::runtime_error("Invalid HNSW binary magic");
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
        M_max0_ = M_ * 2;
        m_L_ = 1.0 / std::log(static_cast<double>(M_));
        in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&enter_point_), sizeof(enter_point_));
        in.read(reinterpret_cast<char*>(&n), sizeof(n));

        ids_.resize(n);
        vectors_.resize(n * dim_);
        in.read(reinterpret_cast<char*>(ids_.data()), n * sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(vectors_.data()), vectors_.size() * sizeof(float));

        uint64_t num_layers = 0;
        in.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
        links_.resize(num_layers);
        for (uint64_t l = 0; l < num_layers; ++l) {
            uint64_t layer_nodes = 0;
            in.read(reinterpret_cast<char*>(&layer_nodes), sizeof(layer_nodes));
            links_[l].resize(layer_nodes);
            for (uint64_t i = 0; i < layer_nodes; ++i) {
                uint32_t deg = 0;
                in.read(reinterpret_cast<char*>(&deg), sizeof(deg));
                links_[l][i].resize(deg);
                in.read(reinterpret_cast<char*>(links_[l][i].data()), deg * sizeof(uint32_t));
            }
        }
    }

    size_t size() const { return ids_.size(); }

private:
    using Pair = std::pair<float, uint32_t>;

    int sample_level() {
        std::uniform_real_distribution<double> dist(1e-9, 1.0);
        return static_cast<int>(-std::log(dist(rng_)) * m_L_);
    }

    const float* get_vec(uint32_t idx) const { return &vectors_[idx * dim_]; }

    // Cosine distance on L2-normalized embeddings
    float distance(const float* a, const float* b) const {
        float dot = 0.0f;
        for (size_t i = 0; i < dim_; ++i) dot += a[i] * b[i];
        return 1.0f - dot;
    }

    std::vector<Pair> search_layer(const float* query, uint32_t ep, size_t ef, int level) const {
        std::unordered_set<uint32_t> visited{ep};
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> candidates; // min-heap
        std::priority_queue<Pair> top_results;                                     // max-heap

        float d0 = distance(query, get_vec(ep));
        candidates.emplace(d0, ep);
        top_results.emplace(d0, ep);

        while (!candidates.empty()) {
            auto [c_dist, c_id] = candidates.top();
            candidates.pop();
            if (c_dist > top_results.top().first && top_results.size() >= ef) break;

            if (level < static_cast<int>(links_.size()) && c_id < links_[level].size()) {
                for (uint32_t nb : links_[level][c_id]) {
                    if (visited.insert(nb).second) {
                        float d = distance(query, get_vec(nb));
                        if (top_results.size() < ef || d < top_results.top().first) {
                            candidates.emplace(d, nb);
                            top_results.emplace(d, nb);
                            if (top_results.size() > ef) top_results.pop();
                        }
                    }
                }
            }
        }

        std::vector<Pair> sorted;
        sorted.reserve(top_results.size());
        while (!top_results.empty()) {
            sorted.push_back(top_results.top());
            top_results.pop();
        }
        std::reverse(sorted.begin(), sorted.end());
        return sorted;
    }

    static std::vector<uint32_t> select_neighbors(const std::vector<Pair>& sorted_cands, size_t M) {
        size_t take = std::min(M, sorted_cands.size());
        std::vector<uint32_t> res(take);
        for (size_t i = 0; i < take; ++i) res[i] = sorted_cands[i].second;
        return res;
    }

    size_t dim_, M_, M_max0_, ef_construction_;
    double m_L_;
    int max_level_ = 0;
    uint32_t enter_point_ = 0;
    std::mt19937 rng_;
    std::vector<float> vectors_;
    std::vector<uint32_t> ids_;
    std::vector<std::vector<std::vector<uint32_t>>> links_;
};