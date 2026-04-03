#pragma once
// =============================================================================
//  rl_bridge.h  v3  —  ZeroMQ Bridge  (simulator.cpp compatible)
//
//  KEY CHANGES from v2:
//   1. graph::Vertex_A now has csv_demand (not base_demand), no id field.
//      substations are identified by position in substations_list[].
//   2. Action space expanded to 4:
//        0 = NO_ACTION
//        1 = REROUTE_WIDEST_PATH
//        2 = THROTTLE_10PCT
//        3 = THROTTLE_20PCT
//   3. Reward functions completely rewritten per spec:
//        Action 0: +2*(1-avg_util) if stable,  -10*(max_util-1)^2 if overloaded
//        Action 1: sum of (util_old-util_new) on previously-over-100% edges - 0.1
//        Action 2: -(50*power_shed/total_demand) - 1.0
//        Action 3: -(50*power_shed/total_demand) - 4.0
//        Cascade:  -100 * (8760 - current_hour) / 8760  (time-dependent)
//   4. Overload threshold is 0.90 (matches simulator.cpp).
//   5. substations_list is built lazily inside poll() on first call.
//   6. update_demands_from_csv uses csv_demand (not base_demand).
// =============================================================================

#include <bits/stdc++.h>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using json = nlohmann::json;

// ---------------------------------------------------------------------------
//  Action enum — 4 discrete actions
// ---------------------------------------------------------------------------
enum class RLAction : int {
    NO_ACTION        = 0,
    REROUTE          = 1,
    THROTTLE_10PCT   = 2,
    THROTTLE_20PCT   = 3
};

// ---------------------------------------------------------------------------
//  ObsLayout
// ---------------------------------------------------------------------------
struct ObsLayout {
    int n_edges = 0;
    int n_subs  = 50;
    int n_time  = 8;
    int total() const { return n_edges + n_subs + n_time; }
};

// ---------------------------------------------------------------------------
//  StepResult
// ---------------------------------------------------------------------------
struct StepResult {
    std::vector<float> obs;
    float reward        = 0.f;
    bool  done          = false;
    int   n_overloaded  = 0;      // edges >= 90%
    float max_severity  = 0.f;    // worst edge ratio
    float avg_util      = 0.f;    // mean edge utilisation
    float total_demand  = 0.f;    // sum of all substation demands
    float power_shed    = 0.f;    // MW cut by throttling this step
    int   sim_hour      = 0;
    int   sim_month     = 0;
    int   csv_row       = 0;
    bool  cascading     = false;
};

// ---------------------------------------------------------------------------
//  RewardCalculator  — implements all 4 reward formulas exactly as specified
// ---------------------------------------------------------------------------
class RewardCalculator {
public:
    static constexpr float CASCADE_FRAC     = 0.95f;   // 95% edges overloaded = cascade
    static constexpr float K_SHED           = 50.0f;   // scaling factor for throttle penalty
    static constexpr float THROTTLE_10_BASE = 1.0f;    // base cost Action 2
    static constexpr float THROTTLE_20_BASE = 4.0f;    // base cost Action 3
    static constexpr float REROUTE_COST     = 0.1f;    // switching wear cost
    static constexpr float OVERLOAD_THRESH  = 0.90f;   // matches simulator.cpp

    // ── Action 0: No Action ────────────────────────────────────────────────
    //  stable:    R = +2.0 * (1.0 - avg_util)
    //  overloaded: R = -10.0 * (max_util - 1.0)^2
    static float action0(float avg_util, float max_util, int n_overloaded) {
        if (n_overloaded == 0) {
            return 2.0f * (1.0f - avg_util);
        } else {
            float excess = std::max(0.f, max_util - 1.0f);
            return -10.0f * excess * excess;
        }
    }

    // ── Action 1: Reroute ─────────────────────────────────────────────────
    //  R = sum_i[ (util_old_i - util_new_i) * I(util_old_i > 1.0) ] - cost_switch
    //  This is computed externally (before/after snapshot) and passed in as delta_reward.
    static float action1(float delta_overload_improvement, bool any_new_overloads,
                         bool was_already_stable) {
        float r = delta_overload_improvement - REROUTE_COST;
        if (any_new_overloads)  r -= 10.0f;    // reroute CAUSED a new overload
        if (was_already_stable) r  = 0.5f;     // grid was fine — small positive
        return r;
    }

    // ── Action 2 & 3: Throttle ────────────────────────────────────────────
    //  R = -(K * power_shed / total_demand) - base_cost
    static float throttle(float power_shed, float total_demand, float base_cost) {
        float ue_fraction = (total_demand > 0.01f) ? (power_shed / total_demand) : 0.f;
        return -(K_SHED * ue_fraction) - base_cost;
    }

    // ── Cascade penalty: time-dependent ───────────────────────────────────
    //  R = -100 * (8760 - current_hour) / 8760
    static float cascade(int current_hour) {
        float hours_remaining = static_cast<float>(8760 - current_hour);
        return -100.0f * (hours_remaining / 8760.0f);
    }

    // ── Master compute: called after action is applied ─────────────────────
    static float compute(RLAction action,
                         float avg_util, float max_util,
                         int n_overloaded, int total_edges,
                         float delta_overload_improvement, bool caused_new_overload,
                         bool was_stable_before,
                         float power_shed, float total_demand,
                         int csv_row, bool& out_done) {
        out_done = false;
        float r  = 0.f;

        // Check cascade first
        bool cascade_event = (total_edges > 0 &&
                              (float)n_overloaded / total_edges >= CASCADE_FRAC);
        if (cascade_event) {
            r = cascade(csv_row);
            out_done = true;
            return r;
        }

        switch (action) {
            case RLAction::NO_ACTION:
                r = action0(avg_util, max_util, n_overloaded);
                break;

            case RLAction::REROUTE:
                r = action1(delta_overload_improvement, caused_new_overload, was_stable_before);
                break;

            case RLAction::THROTTLE_10PCT:
                r = throttle(power_shed, total_demand, THROTTLE_10_BASE);
                break;

            case RLAction::THROTTLE_20PCT:
                r = throttle(power_shed, total_demand, THROTTLE_20_BASE);
                break;
        }

        return r;
    }
};

// ---------------------------------------------------------------------------
//  RLBridgeServer
// ---------------------------------------------------------------------------
class RLBridgeServer {
public:
    explicit RLBridgeServer(const std::string& endpoint = "tcp://*:5556")
        : ctx_(1), socket_(ctx_, zmq::socket_type::rep), csv_row_(0)
    {
        socket_.set(zmq::sockopt::rcvtimeo, 0);
        socket_.bind(endpoint);
        std::cout << "[Bridge] ZMQ REP bound to " << endpoint << "\n";
    }

    template<typename G_t>
    bool poll(G_t& G,
              const std::vector<std::vector<float>>& demand_schedule,
              int& io_sim_hour, int& io_sim_month,
              int& io_sim_day,  int& io_sim_date,
              ObsLayout& layout,
              StepResult& out_result)
    {
        // Build substations_list once on first call (simulator.cpp doesn't maintain one)
        if (substations_list_.empty()) {
            build_substations_list(G);
            std::cout << "[Bridge] substations_list built: "
                      << substations_list_.size() << " substations\n";
        }

        zmq::message_t msg;
        auto res = socket_.recv(msg, zmq::recv_flags::dontwait);
        if (!res) return false;

        json req  = json::parse(msg.to_string_view());
        std::string cmd = req.value("cmd", "step");
        json reply;

        if (cmd == "ping") {
            // layout.n_edges may be 0 on first ping — trigger a quick obs build
            if (layout.n_edges == 0) build_obs(G, 0, 1, layout);
            reply["status"]  = "ok";
            reply["obs_dim"] = layout.total();
            reply["n_rows"]  = (int)demand_schedule.size();
        }
        else if (cmd == "reset") {
            csv_row_     = 0;
            io_sim_hour  = 0; io_sim_day = 0; io_sim_date = 1; io_sim_month = 1;
            if (!demand_schedule.empty())
                update_demands(G, demand_schedule[0]);
            reply["obs"]    = build_obs(G, 0, 1, layout);
            reply["reward"] = 0.0f;
            reply["done"]   = false;
            reply["info"]   = { {"csv_row", 0} };
        }
        else {   // "step"
            int action_int = req.value("action", 0);
            out_result = execute_step(G, demand_schedule,
                                      io_sim_hour, io_sim_month,
                                      io_sim_day, io_sim_date,
                                      static_cast<RLAction>(action_int), layout);

            reply["obs"]                    = out_result.obs;
            reply["reward"]                 = out_result.reward;
            reply["done"]                   = out_result.done;
            reply["info"]["n_overloaded"]   = out_result.n_overloaded;
            reply["info"]["max_severity"]   = out_result.max_severity;
            reply["info"]["avg_util"]       = out_result.avg_util;
            reply["info"]["total_demand"]   = out_result.total_demand;
            reply["info"]["power_shed"]     = out_result.power_shed;
            reply["info"]["cascading"]      = out_result.cascading;
            reply["info"]["sim_hour"]       = out_result.sim_hour;
            reply["info"]["sim_month"]      = out_result.sim_month;
            reply["info"]["csv_row"]        = out_result.csv_row;
        }

        socket_.send(zmq::buffer(reply.dump()), zmq::send_flags::none);
        return true;
    }

private:
    zmq::context_t ctx_;
    zmq::socket_t  socket_;
    int csv_row_ = 0;

    // Flat list of all substation pointers — built once at startup
    // (simulator.cpp doesn't have substations_list; we maintain it here)
    std::vector<void*> substations_list_;

    // ── Build substations_list from graph adjacency maps ─────────────────
    template<typename G_t>
    void build_substations_list(G_t& G) {
        std::unordered_set<typename G_t::Vertex_A*> seen;
        for (auto& [node, neighbors] : G.adj_power_substation) {
            if (node->node_type == "substation" && !seen.count(node)) {
                seen.insert(node);
                substations_list_.push_back(static_cast<void*>(node));
            }
            for (auto& [child, edge] : neighbors) {
                if (child->node_type == "substation" && !seen.count(child)) {
                    seen.insert(child);
                    substations_list_.push_back(static_cast<void*>(child));
                }
            }
        }
    }

    // ── Update demands from CSV row ───────────────────────────────────────
    //  simulator.cpp uses csv_demand (not base_demand) and no id field.
    //  We update all substations in order of substations_list_.
    template<typename G_t>
    void update_demands(G_t& G, const std::vector<float>& new_demands) {
        using VA = typename G_t::Vertex_A;
        int idx = 0;
        for (void* vp : substations_list_) {
            if (idx >= (int)new_demands.size()) break;
            VA* sub = static_cast<VA*>(vp);
            float new_d = new_demands[idx];
            float delta = new_d - sub->csv_demand;
            if (std::fabs(delta) > 0.01f) {
                sub->csv_demand = new_d;
                sub->current_downstream_demand += delta;
                // Propagate delta up the tree
                std::unordered_set<VA*> vis;
                propagate_up(G, sub, delta, vis);
            }
            ++idx;
        }
    }

    // Propagate a demand delta up to parents (mirrors simulator.cpp logic)
    template<typename G_t>
    using VA_t = typename G_t::Vertex_A;

    template<typename G_t>
    void propagate_up(G_t& G, typename G_t::Vertex_A* node, float delta,
                      std::unordered_set<typename G_t::Vertex_A*>& visited) {
        using VA = typename G_t::Vertex_A;
        if (visited.count(node)) return;
        visited.insert(node);

        if (!G.adj_reverse_power.count(node)) { visited.erase(node); return; }

        float total_cap = 0.f;
        for (auto& [parent, edge] : G.adj_reverse_power[node])
            total_cap += edge->max_load;

        for (auto& [parent, edge] : G.adj_reverse_power[node]) {
            float prop = (total_cap > 0.01f) ? delta * edge->max_load / total_cap
                                              : delta / G.adj_reverse_power[node].size();
            edge->current_load += prop;
            parent->current_downstream_demand += prop;
            propagate_up(G, parent, prop, visited);
        }
        visited.erase(node);
    }

    // ── Execute one step ──────────────────────────────────────────────────
    template<typename G_t>
    StepResult execute_step(G_t& G,
                            const std::vector<std::vector<float>>& demand_schedule,
                            int& io_sim_hour, int& io_sim_month,
                            int& io_sim_day,  int& io_sim_date,
                            RLAction action, ObsLayout& layout)
    {
        using VA = typename G_t::Vertex_A;
        const int total_rows = (int)demand_schedule.size();

        // 1. Advance monotonic counter
        csv_row_++;
        bool year_end = (csv_row_ >= total_rows);
        if (year_end) csv_row_ = total_rows - 1;

        // 2. Derive UI calendar
        {
            int r         = csv_row_;
            io_sim_hour   = r % 24;
            int day_of_yr = r / 24;
            io_sim_month  = day_of_yr / 30 + 1;
            io_sim_date   = day_of_yr % 30 + 1;
            io_sim_day    = (r / 24) % 7;
        }

        // 3. Load new demands
        update_demands(G, demand_schedule[csv_row_]);

        // 4. Snapshot BEFORE action (needed for reroute delta reward)
        std::vector<std::pair<typename G_t::Edge*, float>> pre_utils;
        float total_demand_before = 0.f;
        {
            for (void* vp : substations_list_)
                total_demand_before += static_cast<VA*>(vp)->current_downstream_demand;
            for (auto& [src, neighbors] : G.adj_power_substation)
                for (auto& [dst, edge] : neighbors)
                    pre_utils.push_back({edge, (edge->max_load > 0.01f)
                                               ? edge->current_load / edge->max_load : 0.f});
        }

        // 5. Compute total_demand and power_shed for throttle actions
        float power_shed = 0.f;
        float total_demand = total_demand_before;

        // 6. Apply action
        switch (action) {
            case RLAction::REROUTE:
                G.overloading_edge();
                break;

            case RLAction::THROTTLE_10PCT: {
                // Reduce each substation's csv_demand by 10%
                for (void* vp : substations_list_) {
                    VA* sub = static_cast<VA*>(vp);
                    float cut = sub->csv_demand * 0.10f;
                    if (cut > 0.01f) {
                        power_shed += cut;
                        sub->csv_demand -= cut;
                        sub->current_downstream_demand -= cut;
                        std::unordered_set<VA*> vis;
                        propagate_up(G, sub, -cut, vis);
                    }
                }
                G.overloading_edge();
                G.apply_demand_reduction_updates();
                break;
            }

            case RLAction::THROTTLE_20PCT: {
                for (void* vp : substations_list_) {
                    VA* sub = static_cast<VA*>(vp);
                    float cut = sub->csv_demand * 0.20f;
                    if (cut > 0.01f) {
                        power_shed += cut;
                        sub->csv_demand -= cut;
                        sub->current_downstream_demand -= cut;
                        std::unordered_set<VA*> vis;
                        propagate_up(G, sub, -cut, vis);
                    }
                }
                G.overloading_edge();
                G.apply_demand_reduction_updates();
                break;
            }

            case RLAction::NO_ACTION:
            default:
                G.check_node_overloads();
                break;
        }

        // 7. Count overloads AFTER action, compute util stats
        int   n_over   = 0;
        float max_sev  = 0.f;
        float sum_util = 0.f;
        int   total_e  = 0;

        for (auto& [src, neighbors] : G.adj_power_substation) {
            for (auto& [dst, edge] : neighbors) {
                ++total_e;
                if (edge->max_load > 0.01f) {
                    float ratio = edge->current_load / edge->max_load;
                    sum_util += ratio;
                    if (ratio >= RewardCalculator::OVERLOAD_THRESH) ++n_over;
                    if (ratio > max_sev) max_sev = ratio;
                }
            }
        }
        float avg_util = (total_e > 0) ? sum_util / total_e : 0.f;

        // 8. Reroute delta reward: sum (util_old - util_new) for edges that were >100%
        float delta_overload_improvement = 0.f;
        bool  caused_new_overload        = false;
        bool  was_stable_before          = true;

        for (auto& [edge_ptr, old_util] : pre_utils) {
            if (old_util > RewardCalculator::OVERLOAD_THRESH) was_stable_before = false;
            float new_util = (edge_ptr->max_load > 0.01f)
                             ? edge_ptr->current_load / edge_ptr->max_load : 0.f;
            if (old_util > 1.0f) {
                delta_overload_improvement += (old_util - new_util);
            }
            // New overload caused by rerouting: wasn't overloaded before, is now
            if (old_util < RewardCalculator::OVERLOAD_THRESH &&
                new_util >= RewardCalculator::OVERLOAD_THRESH &&
                action == RLAction::REROUTE) {
                caused_new_overload = true;
            }
        }

        // 9. Reward + done
        StepResult r;
        r.cascading = (total_e > 0 &&
                       (float)n_over / total_e >= RewardCalculator::CASCADE_FRAC);
        r.reward    = RewardCalculator::compute(
            action,
            avg_util, max_sev,
            n_over, total_e,
            delta_overload_improvement, caused_new_overload, was_stable_before,
            power_shed, total_demand,
            csv_row_, r.done);

        if (year_end) r.done = true;

        r.n_overloaded = n_over;
        r.max_severity = max_sev;
        r.avg_util     = avg_util;
        r.total_demand = total_demand;
        r.power_shed   = power_shed;
        r.sim_hour     = io_sim_hour;
        r.sim_month    = io_sim_month;
        r.csv_row      = csv_row_;

        r.obs = build_obs(G, io_sim_hour, io_sim_month, layout);
        return r;
    }

    // ── Observation builder ───────────────────────────────────────────────
    template<typename G_t>
    std::vector<float> build_obs(G_t& G, int hour, int month, ObsLayout& layout) {
        std::vector<float> obs;
        int edge_count = 0;

        for (auto& [src, neighbors] : G.adj_power_substation) {
            for (auto& [dst, edge] : neighbors) {
                float util = (edge->max_load > 0.01f)
                    ? std::clamp(edge->current_load / edge->max_load, 0.f, 2.f) : 0.f;
                obs.push_back(util);
                ++edge_count;
            }
        }
        layout.n_edges = edge_count;

        using VA = typename G_t::Vertex_A;
        for (void* vp : substations_list_) {
            VA* sub = static_cast<VA*>(vp);
            float norm = (sub->max_limit > 0.01f)
                ? std::clamp(sub->current_downstream_demand / sub->max_limit, 0.f, 2.f) : 0.f;
            obs.push_back(norm);
        }

        obs.push_back(sinf(2.f * (float)M_PI * hour  / 24.f));
        obs.push_back(cosf(2.f * (float)M_PI * hour  / 24.f));
        obs.push_back(sinf(2.f * (float)M_PI * month / 12.f));
        obs.push_back(cosf(2.f * (float)M_PI * month / 12.f));
        for (int i = 0; i < 4; ++i) obs.push_back(0.f);

        return obs;
    }
};
