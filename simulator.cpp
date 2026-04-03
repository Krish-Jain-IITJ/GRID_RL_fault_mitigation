    #include <bits/stdc++.h>
    #include <cmath>
    #include <queue>
    #include <unordered_set>
    #include <iomanip>
    #include <fstream>
    #include <sstream>
    #include "raylib.h"
    #include "raymath.h"
    #include "rl_bridge.h"   // ← RL Bridge

    using namespace std;

    static random_device rd;
    static mt19937 g(rd());

    // ── Global simulation clock (mutated by RLBridgeServer::execute_step) ────
    int sim_hour        = 0;
    int sim_day_of_week = 0;
    int sim_date        = 1;
    int sim_month       = 1;

    struct PairHash
    {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    class graph{
    public:
        class Vertex_A
        {
        public:
            string node_type = "";
            float max_limit;
            float csv_demand;                   // current CSV demand value
            float current_downstream_demand;
            int x, y;

            Vertex_A(string type) : node_type(type), max_limit(0),
                                    csv_demand(0), current_downstream_demand(0),
                                    x(0), y(0) {}
        };

        class Edge
        {
        public:
            float loss;
            float max_load;
            float current_load;

            Edge(Vertex_A *Node_1, Vertex_A *Node_2) : loss(0), max_load(0), current_load(0)
            {
                if (Node_1->node_type == "power_plant" && Node_2->node_type == "substation")
                    loss = 2 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (5 - 2)));
                else if (Node_1->node_type == "substation" && Node_2->node_type == "substation")
                    loss = 1 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (3 - 1)));
            };
        };

        unordered_map<Vertex_A *, vector<pair<Vertex_A *, Edge *>>> adj_power_substation;
        unordered_map<Vertex_A *, vector<pair<Vertex_A *, Edge *>>> adj_reverse_power;
        unordered_set<Edge *> overloaded_edges;
        Edge *edge_being_fixed = nullptr;
        unordered_map<Vertex_A *, float> nodes_to_throttle;
        unordered_set<Vertex_A *> node_overloads_visual;
        unordered_set<Vertex_A *> throttled_nodes_visual;
        vector<pair<Vertex_A *, Edge *>> fix_path;
        Vertex_A *last_event_substation = nullptr;
        Color last_event_substation_color = {0, 0, 0, 0};

        // Fixed max-limit table (50 substations, from original simulator.cpp)
        vector<float> max_limits = {
            118.0f, 51.0f, 33.0f, 108.0f, 21.0f, 118.0f, 26.0f, 35.0f, 72.0f, 13.0f,
              9.0f, 56.0f, 22.0f,  70.0f,  8.0f,  10.0f,115.0f, 80.0f, 27.0f, 15.0f,
             71.0f,114.0f, 91.0f,  71.0f, 10.0f,  82.0f, 90.0f, 25.0f, 36.0f, 50.0f,
             45.0f, 35.0f, 73.0f, 101.0f,120.0f,  25.0f,100.0f, 60.0f, 17.0f, 29.0f,
             67.0f, 23.0f, 50.0f,  55.0f, 54.0f,  91.0f, 50.0f, 46.0f, 15.0f, 28.0f
        };

        // ── CSV / demand helpers ────────────────────────────────────────────
        //  Full-year CSV loader: returns demand_schedule[row][sub_idx]
        vector<vector<float>> load_full_year_csv(const string& filename) {
            vector<vector<float>> schedule;
            ifstream file(filename);
            if (!file.is_open()) {
                cerr << "[CSV] Cannot open '" << filename << "'\n";
                return schedule;
            }
            string line;
            getline(file, line); // skip header
            while (getline(file, line)) {
                vector<float> row;
                stringstream ss(line);
                string token;
                getline(ss, token, ','); // skip datetime
                while (getline(ss, token, ',')) {
                    try { row.push_back(stof(token)); }
                    catch (...) { row.push_back(0.f); }
                }
                schedule.push_back(row);
            }
            cout << "[CSV] Loaded " << schedule.size() << " rows × "
                 << (schedule.empty() ? 0 : schedule[0].size()) << " columns\n";
            return schedule;
        }

        //  Old single-row reader kept for backward compatibility
        vector<float> read_demand_csv(const string &filename) {
            auto schedule = load_full_year_csv(filename);
            return schedule.empty() ? vector<float>() : schedule[0];
        }

        void mapping(int powerplant_count, int substation_count)
        {
            vector<float> csv_demands = read_demand_csv("smooth_substation_power_demand_1yr.csv");
            const float DEFAULT_DEMAND = 50.0f;

            vector<Vertex_A *> powerplants;
            for (int i = 0; i < powerplant_count; i++)
                powerplants.push_back(new Vertex_A("power_plant"));

            vector<Vertex_A *> substations;
            for (int i = 0; i < substation_count; i++) {
                Vertex_A *s = new Vertex_A("substation");
                float demand = (i < (int)csv_demands.size()) ? csv_demands[i] : DEFAULT_DEMAND;
                set_substation_demand(s, demand, i);
                substations.push_back(s);
            }
            map_powerplants_to_substations(powerplants, substations);
        }

        void set_substation_demand(Vertex_A *substation, float demand, int i)
        {
            substation->max_limit = (i < (int)max_limits.size()) ? max_limits[i] : demand * 1.5f;
            substation->current_downstream_demand = demand;
            substation->csv_demand = demand;
        }

        void map_powerplants_to_substations(vector<Vertex_A *> &powerplants,
                                            vector<Vertex_A *> &substations)
        {
            int num_powerplants = (int)powerplants.size();
            int num_substations = (int)substations.size();
            int substations_for_powerplants = max(1, (int)(0.4 * num_substations));

            vector<int> substation_indices(num_substations);
            iota(substation_indices.begin(), substation_indices.end(), 0);
            shuffle(substation_indices.begin(), substation_indices.end(), g);
            int idx = 0;

            for (int i = 0; i < substations_for_powerplants && idx < num_substations; i++, idx++) {
                Vertex_A *plant = powerplants[i % num_powerplants];
                Vertex_A *sub   = substations[substation_indices[idx]];
                Edge *edge      = new Edge(plant, sub);
                edge->current_load = sub->current_downstream_demand;
                edge->max_load     = 2 * sub->max_limit;
                adj_power_substation[plant].push_back({sub, edge});
                adj_reverse_power[sub].push_back({plant, edge});
            }

            while (idx < num_substations) {
                if (idx == 0) break;
                int src_idx = rand() % idx;
                Vertex_A *src  = substations[substation_indices[src_idx]];
                Vertex_A *dest = substations[substation_indices[idx]];
                Edge *edge     = new Edge(src, dest);
                src->current_downstream_demand += dest->current_downstream_demand;
                edge->current_load = dest->current_downstream_demand;
                edge->max_load     = dest->max_limit;
                adj_power_substation[src].push_back({dest, edge});
                adj_reverse_power[dest].push_back({src, edge});
                idx++;
            }

            for (auto &plant_entry : adj_power_substation) {
                Vertex_A *plant = plant_entry.first;
                if (plant->node_type != "power_plant") continue;
                float sd = 0;
                for (auto &p : plant_entry.second) sd += p.first->current_downstream_demand;
                plant->max_limit = 0;
                plant->current_downstream_demand = sd;
            }
            compute_capacity_backtracking_bfs();
            assign_edge_max_load_bfs();
        }

    private:
        void compute_capacity_backtracking_bfs()
        {
            queue<Vertex_A*> q;
            vector<Vertex_A*> bfs_order;
            unordered_set<Vertex_A*> visited;

            for (auto &[node, _] : adj_power_substation)
                if (node->node_type == "power_plant") { q.push(node); visited.insert(node); }

            while (!q.empty()) {
                Vertex_A* curr = q.front(); q.pop();
                bfs_order.push_back(curr);
                if (!adj_power_substation.count(curr)) continue;
                for (auto &[child, edge] : adj_power_substation[curr])
                    if (!visited.count(child)) { visited.insert(child); q.push(child); }
            }

            for (int i = (int)bfs_order.size() - 1; i >= 0; i--) {
                Vertex_A* node = bfs_order[i];
                if (!adj_power_substation.count(node)) continue;
                float total = node->max_limit;
                for (auto &[child, edge] : adj_power_substation[node]) total += child->max_limit;
                node->max_limit = total;
            }
        }

        void assign_edge_max_load_bfs()
        {
            queue<Vertex_A*> q;
            unordered_set<Vertex_A*> visited;
            for (auto &[node, _] : adj_power_substation)
                if (node->node_type == "power_plant") { q.push(node); visited.insert(node); }

            while (!q.empty()) {
                Vertex_A* parent = q.front(); q.pop();
                if (!adj_power_substation.count(parent)) continue;
                for (auto &[child, edge] : adj_power_substation[parent]) {
                    edge->max_load = child->max_limit * 0.75f;
                    if (!visited.count(child)) { visited.insert(child); q.push(child); }
                }
            }
        }

        double GetDistance(int x1, int y1, int x2, int y2)
        { return sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0)); }
        int GetRandom(int a, int b) {
            if (a > b) swap(a,b);
            return (rand()%(b-a+1))+a;
        }
        using GridKey = pair<int,int>;

        bool is_position_ok_A(Vertex_A *n, int x, int y,
                              const unordered_map<GridKey,vector<Vertex_A*>,PairHash>& grid, int CS) {
            float md = (n->node_type=="power_plant") ? 400.f : 50.f;
            int gx=x/CS, gy=y/CS;
            for (int dx=-1;dx<=1;dx++) for (int dy=-1;dy<=1;dy++) {
                GridKey k={gx+dx,gy+dy};
                if (grid.count(k)) for (auto* o : grid.at(k))
                    if (o->node_type==n->node_type && GetDistance(x,y,o->x,o->y)<md) return false;
            }
            return true;
        }

        bool DoesEdgeExist(Vertex_A *a, Vertex_A *b) {
            auto check=[&](Vertex_A* s,Vertex_A* t){
                if(adj_power_substation.count(s)) for(auto& p:adj_power_substation[s]) if(p.first==t) return true;
                return false;
            };
            return check(a,b)||check(b,a);
        }

        struct PathState {
            float cap; vector<pair<Vertex_A*,Edge*>> path;
            PathState(float c,vector<pair<Vertex_A*,Edge*>> p):cap(c),path(move(p)){}
            bool operator<(const PathState& o)const{return cap<o.cap;}
        };

        pair<float,vector<pair<Vertex_A*,Edge*>>> find_highest_capacity_path(Vertex_A* src,Vertex_A* tgt) {
            priority_queue<PathState> pq;
            pq.push(PathState(numeric_limits<float>::max(),{{src,nullptr}}));
            unordered_map<Vertex_A*,float> best; best[src]=numeric_limits<float>::max();
            float bCap=0; vector<pair<Vertex_A*,Edge*>> bPath;
            while (!pq.empty()) {
                auto cur=pq.top(); pq.pop();
                auto* cn=cur.path.back().first;
                if(best.count(cn)&&cur.cap<best[cn]) continue;
                if(cn==tgt){if(cur.cap>bCap){bCap=cur.cap;bPath=cur.path;}continue;}
                auto expand=[&](Vertex_A* nb,Edge* e){
                    for(auto& p:cur.path) if(p.first==nb) return;
                    float nc=min(cur.cap,e->max_load-e->current_load);
                    if(nc<=0.01f) return;
                    if(!best.count(nb)||nc>best[nb]){
                        best[nb]=nc; auto np=cur.path; np.push_back({nb,e});
                        pq.push(PathState(nc,np));
                    }
                };
                if(adj_power_substation.count(cn)) for(auto&[nb,e]:adj_power_substation[cn]) expand(nb,e);
                if(adj_reverse_power.count(cn))    for(auto&[nb,e]:adj_reverse_power[cn])    expand(nb,e);
            }
            return {bCap,bPath};
        }

        void try_add_capacity_path(const PathState& cur, Vertex_A* nb, Edge* e,
                                   priority_queue<PathState>& pq, unordered_map<Vertex_A*,float>& best) {
            for(auto& p:cur.path) if(p.first==nb) return;
            float nc=min(cur.cap,e->max_load-e->current_load);
            if(nc<=0.01f) return;
            if(!best.count(nb)||nc>best[nb]){
                best[nb]=nc; auto np=cur.path; np.push_back({nb,e});
                pq.push(PathState(nc,np));
            }
        }

        void propagate_demand_change(Vertex_A *node, float delta, unordered_set<Vertex_A *> &vis) {
            if(vis.count(node)) return; vis.insert(node);
            node->current_downstream_demand += delta;
            if(!adj_reverse_power.count(node)){vis.erase(node);return;}
            float tot=0;
            for(auto&[p,e]:adj_reverse_power[node]) tot+=e->max_load;
            for(auto&[p,e]:adj_reverse_power[node]) {
                float pd=(tot>0.01f)?(delta*e->max_load/tot):(delta/adj_reverse_power[node].size());
                e->current_load+=pd;
                propagate_demand_change(p,pd,vis);
            }
            vis.erase(node);
        }

        void propagate_limit_increase(Vertex_A *node, float inc, unordered_set<Vertex_A *>& vis) {
            if(vis.count(node)) return; vis.insert(node);
            if(!adj_reverse_power.count(node)){vis.erase(node);return;}
            for(auto&[p,e]:adj_reverse_power[node]) {
                float pi=(p->node_type=="power_plant")?inc*1.2f:inc;
                p->max_limit+=pi; e->max_load+=inc;
                propagate_limit_increase(p,pi,vis);
            }
            vis.erase(node);
        }

    public:
        vector<pair<string,pair<int,int>>> layout_graph() {
            vector<pair<string,pair<int,int>>> coords;
            unordered_set<Vertex_A*> vis;
            queue<Vertex_A*> q;
            vector<Vertex_A*> plants;
            unordered_map<GridKey,vector<Vertex_A*>,PairHash> grid;
            const int CS=50, RANGE=200;

            for(auto&[n,_]:adj_power_substation) if(n->node_type=="power_plant") plants.push_back(n);

            for(auto* pl:plants){
                if(vis.count(pl)) continue;
                bool placed=false;
                for(int t=0;t<1000;t++){
                    int x=GetRandom(0,2000),y=GetRandom(0,2000);
                    if(is_position_ok_A(pl,x,y,grid,CS)){
                        pl->x=x;pl->y=y;vis.insert(pl);q.push(pl);
                        coords.push_back({"power_plant",{x,y}});
                        grid[{x/CS,y/CS}].push_back(pl); placed=true; break;
                    }
                }
                if(!placed){
                    pl->x=GetRandom(0,2000);pl->y=GetRandom(0,2000);
                    vis.insert(pl);q.push(pl);coords.push_back({"power_plant",{pl->x,pl->y}});
                    grid[{pl->x/CS,pl->y/CS}].push_back(pl);
                }
            }

            while(!q.empty()){
                Vertex_A* par=q.front();q.pop();
                if(!adj_power_substation.count(par)) continue;
                for(auto&[ch,_]:adj_power_substation[par]){
                    if(ch->node_type!="substation"||vis.count(ch)) continue;
                    bool placed=false;
                    for(int t=0;t<1000;t++){
                        int x=GetRandom(par->x-RANGE,par->x+RANGE);
                        int y=GetRandom(par->y-RANGE,par->y+RANGE);
                        if(is_position_ok_A(ch,x,y,grid,CS)){
                            ch->x=x;ch->y=y;vis.insert(ch);q.push(ch);
                            coords.push_back({"substation",{x,y}});
                            grid[{x/CS,y/CS}].push_back(ch); placed=true; break;
                        }
                    }
                    if(!placed){
                        ch->x=par->x;ch->y=par->y;vis.insert(ch);q.push(ch);
                        coords.push_back({"substation",{ch->x,ch->y}});
                        grid[{ch->x/CS,ch->y/CS}].push_back(ch);
                    }
                }
            }
            return coords;
        }

        void add_realistic_connections(int k_nearest=2) {
            unordered_set<Vertex_A*> sub_set;
            for(auto&entry:adj_power_substation){
                if(entry.first->node_type=="substation") sub_set.insert(entry.first);
                for(auto&p:entry.second) if(p.first->node_type=="substation") sub_set.insert(p.first);
            }
            vector<Vertex_A*> subs(sub_set.begin(),sub_set.end());
            if((int)subs.size()<k_nearest+1) return;
            for(auto* s1:subs){
                priority_queue<pair<double,Vertex_A*>> pq;
                for(auto* s2:subs){
                    if(s1==s2) continue;
                    double d=GetDistance(s1->x,s1->y,s2->x,s2->y);
                    if((int)pq.size()<k_nearest) pq.push({d,s2});
                    else if(d<pq.top().first){pq.pop();pq.push({d,s2});}
                }
                while(!pq.empty()){
                    auto* s2=pq.top().second; pq.pop();
                    if(!DoesEdgeExist(s1,s2)){
                        Edge* e=new Edge(s1,s2);
                        e->current_load=0;
                        e->max_load=min(s1->max_limit,s2->max_limit);
                        adj_power_substation[s1].push_back({s2,e});
                        adj_reverse_power[s2].push_back({s1,e});
                    }
                }
            }
        }

        Vertex_A *increase_random_substation_demand() {
            unordered_set<Vertex_A*> sub_set;
            for(auto&e:adj_power_substation){
                if(e.first->node_type=="substation") sub_set.insert(e.first);
                for(auto&p:e.second) if(p.first->node_type=="substation") sub_set.insert(p.first);
            }
            if(sub_set.empty()) return nullptr;
            vector<Vertex_A*> subs(sub_set.begin(),sub_set.end());
            Vertex_A* sub=subs[rand()%subs.size()];
            float increase=max(5.0f,sub->current_downstream_demand*0.2f);
            sub->current_downstream_demand+=increase;
            unordered_set<Vertex_A*> vis;
            propagate_demand_change(sub,increase,vis);
            last_event_substation=sub;
            return sub;
        }

        struct OverloadInfo {
            float ratio; Vertex_A* source; Vertex_A* target; Edge* edge;
            bool operator>(const OverloadInfo& o)const{return ratio>o.ratio;}
        };

        void overloading_edge() {
            vector<OverloadInfo> probs;
            overloaded_edges.clear(); edge_being_fixed=nullptr;
            fix_path.clear(); throttled_nodes_visual.clear();
            for(auto&[s,nb]:adj_power_substation)
                for(auto&[t,e]:nb)
                    if(e->max_load>0.01f){
                        float r=e->current_load/e->max_load;
                        if(r>=0.90f){probs.push_back({r,s,t,e});overloaded_edges.insert(e);}
                    }
            if(probs.empty()) return;
            sort(probs.begin(),probs.end(),greater<OverloadInfo>());
            bool fixed=false;
            for(auto& p:probs)
                if(overloaded_edges.count(p.edge))
                    if(fix_overloading_problem(p.source,p.target,p.edge,!fixed)) fixed=true;
        }

        bool fix_overloading_problem(Vertex_A* src,Vertex_A* tgt,Edge* oe,bool viz) {
            if(!oe) return false;
            const float OT=0.90f, TF=0.80f;
            float cf=(oe->max_load>0)?oe->current_load/oe->max_load:1.f;
            float red=oe->current_load-(TF*oe->max_load);
            if(cf<OT||red<0.01f){if(cf<OT)overloaded_edges.erase(oe);return false;}
            auto[cap,path]=find_highest_capacity_path(src,tgt);
            bool usable=false; float actual=0;
            if(!path.empty()&&path.size()>1){
                actual=min(red,cap);
                if(actual>=(red*0.20f)){
                    bool stable=true;
                    for(size_t i=1;i<path.size();i++){
                        auto* pe=path[i].second;
                        if(pe->max_load>0.01f&&(pe->current_load+actual)/pe->max_load>=OT){stable=false;break;}
                    }
                    if(stable) usable=true;
                }
            }
            if(usable){
                oe->current_load-=actual;
                for(size_t i=1;i<path.size();i++) path[i].second->current_load+=actual;
                if(oe->current_load/oe->max_load<OT) overloaded_edges.erase(oe);
                if(viz){edge_being_fixed=oe;fix_path=path;}
                return true;
            } else {
                overloaded_edges.erase(oe);
                nodes_to_throttle[tgt]+=red;
                return false;
            }
        }

        void apply_demand_reduction_updates() {
            if(nodes_to_throttle.empty()) return;
            for(auto&[sub,req]:nodes_to_throttle){
                if(req<=0.01f) continue;
                float act=-min(req,sub->current_downstream_demand);
                if(act<-0.01f){
                    sub->current_downstream_demand+=act;
                    throttled_nodes_visual.insert(sub);
                    unordered_set<Vertex_A*> vis;
                    propagate_demand_change(sub,act,vis);
                }
            }
            nodes_to_throttle.clear();
        }

        void check_node_overloads() {
            node_overloads_visual.clear();
            unordered_set<Vertex_A*> all;
            for(auto&[p,ch]:adj_power_substation){
                all.insert(p);
                for(auto&pp:ch) all.insert(pp.first);
            }
            for(auto* n:all)
                if(n->node_type=="substation"&&n->current_downstream_demand>n->max_limit)
                    node_overloads_visual.insert(n);
        }

        void upgrade_selected_node_limit(Vertex_A* n) {
            if(!n||n->node_type=="power_plant") return;
            if(node_overloads_visual.count(n)){
                float inc=n->max_limit*0.25f;
                float def=n->current_downstream_demand-n->max_limit;
                if(inc<def) inc=def*1.1f;
                n->max_limit+=inc;
                unordered_set<Vertex_A*> vis;
                propagate_limit_increase(n,inc,vis);
                check_node_overloads();
            }
        }
    };

    // ── Raylib helpers ────────────────────────────────────────────────────────
    void DrawNodeA(graph::Vertex_A *node) {
        if(node->node_type=="power_plant"){DrawCircle(node->x,node->y,15,PINK);DrawCircleLines(node->x,node->y,15,MAROON);}
        else{DrawCircle(node->x,node->y,10,BLUE);DrawCircleLines(node->x,node->y,10,DARKBLUE);}
    }

    void DrawArrow(Vector2 s,Vector2 e,float r,float th,Color c) {
        Vector2 d=Vector2Normalize(Vector2Subtract(e,s));
        Vector2 ap=Vector2Subtract(e,Vector2Scale(d,r));
        DrawLineEx(s,ap,th,c);
        float as=8;
        DrawLineEx(ap,Vector2Add(ap,Vector2Scale(Vector2Rotate(d,-150*DEG2RAD),as/(th>2?1.5f:1.f))),th,c);
        DrawLineEx(ap,Vector2Add(ap,Vector2Scale(Vector2Rotate(d, 150*DEG2RAD),as/(th>2?1.5f:1.f))),th,c);
    }

    enum GameState { STATE_INPUT, STATE_VISUALIZATION };

    int main()
    {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
        srand((unsigned int)time(0));

        const int W=1280, H=720;
        InitWindow(W, H, "Power Grid — DQN Bridge");

        GameState currentState = STATE_INPUT;
        const int inputFieldCount = 2;
        const char* labels[2] = {"Power Plants:", "Substations:"};
        Rectangle textBoxes[2];
        std::string inputStrings[2] = {"3","50"};
        int activeTextBox = -1;

        Rectangle startButton;
        bool showCursor=false; int framesCounter=0;
        int startY=200,inputH=40,inputW=200,labelW=250,padding=15,fSize=20;
        for(int i=0;i<2;i++)
            textBoxes[i]={(float)W/2-inputW/2,(float)startY+i*(inputH+padding),(float)inputW,(float)inputH};
        startButton={(float)W/2-inputW/2,(float)startY+2*(inputH+padding)+20,(float)inputW,(float)inputH+10};

        graph G;
        vector<vector<float>> demand_schedule;

        // ZMQ bridge (created after graph is built)
        RLBridgeServer* bridge = nullptr;
        ObsLayout obs_layout;
        StepResult last_result;

        Camera2D camera={};
        const float ZOOM_LOD=0.5f;
        graph::Vertex_A* sel_node=nullptr;
        graph::Edge* sel_edge=nullptr;
        unordered_set<void*> vis_nodes;
        vector<graph::Vertex_A*> draw_nodes;

        enum FVS{VIZ_IDLE,VIZ_OVERLOAD,VIZ_FIX};
        FVS fvs=VIZ_IDLE; double fvt=0;
        bool rl_mode=true;

        SetTargetFPS(60);

        while(!WindowShouldClose())
        {
            switch(currentState)
            {
            // ── INPUT STATE ────────────────────────────────────────────────
            case STATE_INPUT:
            {
                framesCounter++; showCursor=((framesCounter/30)%2==0);
                if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
                    activeTextBox=-1;
                    for(int i=0;i<2;i++)
                        if(CheckCollisionPointRec(GetMousePosition(),textBoxes[i])){activeTextBox=i;framesCounter=0;break;}
                }
                if(activeTextBox!=-1){
                    int key=GetCharPressed();
                    while(key>0){
                        if(key>='0'&&key<='9'&&inputStrings[activeTextBox].length()<9)
                            inputStrings[activeTextBox]+=(char)key;
                        key=GetCharPressed();
                    }
                    if(IsKeyPressed(KEY_BACKSPACE)&&!inputStrings[activeTextBox].empty())
                        inputStrings[activeTextBox].pop_back();
                }

                if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)&&CheckCollisionPointRec(GetMousePosition(),startButton)){
                    try {
                        int np=stoi(inputStrings[0]), ns=stoi(inputStrings[1]);
                        if(np>0&&ns>0){
                            cout<<"Building graph...\n";
                            G.mapping(np,ns);
                            G.layout_graph();
                            G.add_realistic_connections(2);
                            cout<<"Network built.\n";

                            // Load full-year CSV
                            demand_schedule=G.load_full_year_csv("smooth_substation_power_demand_1yr.csv");

                            // Start ZMQ bridge
                            bridge=new RLBridgeServer("tcp://*:5556");
                            cout<<"[Bridge] Ready. Start Python training.\n";

                            camera.target={(1000,1000)};
                            camera.offset={(float)W/2,(float)H/2};
                            camera.zoom=0.25f;
                            currentState=STATE_VISUALIZATION;
                        }
                    } catch(...){ cout<<"Invalid input\n"; }
                }

                BeginDrawing(); ClearBackground(DARKGRAY);
                DrawText("Power Grid DQN Setup",W/2-MeasureText("Power Grid DQN Setup",30)/2,30,30,WHITE);
                DrawText("CSV: smooth_substation_power_demand_1yr.csv",
                         W/2-MeasureText("CSV: smooth_substation_power_demand_1yr.csv",18)/2,80,18,LIGHTGRAY);
                for(int i=0;i<2;i++){
                    DrawText(labels[i],(int)(textBoxes[i].x-labelW),(int)(textBoxes[i].y+(inputH-fSize)/2.f),fSize,LIGHTGRAY);
                    DrawRectangleRec(textBoxes[i],LIGHTGRAY);
                    DrawRectangleLinesEx(textBoxes[i],activeTextBox==i?2:1,activeTextBox==i?RED:DARKGRAY);
                    DrawText(inputStrings[i].c_str(),(int)(textBoxes[i].x+5),(int)(textBoxes[i].y+(inputH-fSize)/2.f),fSize,BLACK);
                    if(activeTextBox==i&&showCursor){
                        int tw=MeasureText(inputStrings[i].c_str(),fSize);
                        DrawRectangle((int)(textBoxes[i].x+5+tw),(int)(textBoxes[i].y+4),2,inputH-8,MAROON);
                    }
                }
                DrawRectangleRec(startButton,MAROON);
                DrawText("START SIMULATION",(int)(startButton.x+startButton.width/2-MeasureText("START SIMULATION",fSize)/2),
                         (int)(startButton.y+(startButton.height-fSize)/2),fSize,WHITE);
                EndDrawing();
            } break;

            // ── VISUALIZATION STATE ────────────────────────────────────────
            case STATE_VISUALIZATION:
            {
                // Non-blocking ZMQ poll every frame
                if(bridge)
                    bridge->poll(G, demand_schedule,
                                 sim_hour, sim_month, sim_day_of_week, sim_date,
                                 obs_layout, last_result);

                // Rebuild draw list
                vis_nodes.clear(); draw_nodes.clear();
                for(auto&[p,nb]:G.adj_power_substation){
                    if(!vis_nodes.count(p)){draw_nodes.push_back(p);vis_nodes.insert(p);}
                    for(auto&[c,e]:nb) if(!vis_nodes.count(c)){draw_nodes.push_back(c);vis_nodes.insert(c);}
                }

                if(!rl_mode) G.apply_demand_reduction_updates();

                // Camera
                camera.zoom=clamp(camera.zoom+(float)GetMouseWheelMove()*0.05f,0.1f,3.f);
                if(IsMouseButtonDown(MOUSE_BUTTON_LEFT))
                    camera.target=Vector2Add(camera.target,Vector2Scale(GetMouseDelta(),-1.f/camera.zoom));

                // Keys
                if(IsKeyPressed(KEY_R)) rl_mode=!rl_mode;
                if(!rl_mode){
                    if(IsKeyPressed(KEY_I)){
                        G.increase_random_substation_demand();
                        G.check_node_overloads();
                    }
                    if(IsKeyPressed(KEY_O)){
                        G.overloading_edge(); G.check_node_overloads();
                        if(G.edge_being_fixed){fvs=VIZ_OVERLOAD;fvt=GetTime();}
                        G.apply_demand_reduction_updates();
                    }
                    if(IsKeyPressed(KEY_U)) G.upgrade_selected_node_limit(sel_node);
                }
                if(IsKeyPressed(KEY_C)){
                    G.overloaded_edges.clear(); G.edge_being_fixed=nullptr;
                    G.fix_path.clear(); fvs=VIZ_IDLE;
                    G.throttled_nodes_visual.clear(); G.node_overloads_visual.clear();
                }

                // Selection
                if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
                    Vector2 wm=GetScreenToWorld2D(GetMousePosition(),camera);
                    sel_node=nullptr; sel_edge=nullptr; bool hit=false;
                    for(auto* n:draw_nodes){
                        float r=(n->node_type=="power_plant")?15.f:10.f;
                        if(CheckCollisionPointCircle(wm,{(float)n->x,(float)n->y},r)){sel_node=n;hit=true;break;}
                    }
                    if(!hit) for(auto&[p,nb]:G.adj_power_substation)
                        for(auto&[c,e]:nb)
                            if(CheckCollisionPointLine(wm,{(float)p->x,(float)p->y},
                               {(float)c->x,(float)c->y},5/camera.zoom)){sel_edge=e;hit=true;break;}
                }
                if(fvs==VIZ_OVERLOAD&&GetTime()-fvt>=0.5) fvs=VIZ_FIX;

                // ── DRAW ──────────────────────────────────────────────────
                BeginDrawing(); ClearBackground(DARKGRAY);
                BeginMode2D(camera);

                unordered_set<graph::Edge*> path_edges;
                if(fvs==VIZ_FIX) for(size_t i=1;i<G.fix_path.size();i++)
                    if(G.fix_path[i].second) path_edges.insert(G.fix_path[i].second);

                for(auto&[par,nb]:G.adj_power_substation)
                    for(auto&[chi,e]:nb){
                        float ratio=(e->max_load>0.01f)?e->current_load/e->max_load:0.f;
                        Color ec=(ratio>=0.90f)?RED:(ratio>=0.70f)?ORANGE:(ratio>=0.40f)?GREEN:
                                 (ratio>=0.10f)?BLUE:WHITE;
                        Vector2 sv={(float)par->x,(float)par->y};
                        Vector2 ev={(float)chi->x,(float)chi->y};
                        float er=(chi->node_type=="power_plant")?15.f:10.f;

                        bool gfx=(fvs==VIZ_FIX)&&(e==G.edge_being_fixed||path_edges.count(e));
                        bool ofx=(fvs==VIZ_OVERLOAD)&&e==G.edge_being_fixed;
                        bool red=G.overloaded_edges.count(e);

                        if(camera.zoom>=ZOOM_LOD){
                            if(gfx)       DrawArrow(sv,ev,er,4/camera.zoom,GREEN);
                            else if(ofx)  DrawArrow(sv,ev,er,4/camera.zoom,ORANGE);
                            else if(red)  {float th=3.f+sin(GetTime()*10)*1.5f;DrawArrow(sv,ev,er,th/camera.zoom,RED);}
                            else          DrawLineEx(sv,ev,1/camera.zoom,ec);
                            if(e==sel_edge) DrawLineEx(sv,ev,3/camera.zoom,YELLOW);
                        }
                    }

                for(auto* n:draw_nodes){
                    if(n->node_type=="substation"&&camera.zoom<ZOOM_LOD) continue;
                    DrawNodeA(n);
                    if(G.throttled_nodes_visual.count(n)){DrawCircle(n->x,n->y,10,RAYWHITE);DrawCircleLines(n->x,n->y,10,WHITE);}
                    if(G.node_overloads_visual.count(n)) {DrawCircle(n->x,n->y,10,BLACK);DrawCircleLines(n->x,n->y,10,DARKGRAY);}
                    if(n==G.last_event_substation){
                        float pr=10.f+2.f+sin(GetTime()*10)*1.5f;
                        DrawRingLines({(float)n->x,(float)n->y},pr-2/camera.zoom,pr,0,360,36,Fade(YELLOW,0.8f));
                    }
                    if(n==sel_node) DrawCircleLines(n->x,n->y,12,YELLOW);
                }
                EndMode2D();

                // UI overlay
                DrawRectangle(0,0,W,30,Fade(BLACK,0.8f));
                string mode_str=rl_mode?"[RL MODE — Python DQN Driving]":"[MANUAL MODE — Press R to enable RL]";
                DrawText(mode_str.c_str(),10,5,20,rl_mode?GREEN:ORANGE);
                DrawFPS(W-80,5);

                // Clock
                string day_names[]={"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
                string clk="Sim: "+day_names[sim_day_of_week]+", "+to_string(sim_date)+"/"+
                           to_string(sim_month)+" | "+(sim_hour<10?"0":"")+to_string(sim_hour)+":00";
                int cw=MeasureText(clk.c_str(),20)+40;
                DrawRectangle(W/2-cw/2,35,cw,35,Fade(BLACK,0.8f));
                DrawText(clk.c_str(),W/2-MeasureText(clk.c_str(),20)/2,43,20,GREEN);

                // RL stats
                if(rl_mode){
                    DrawRectangle(W-270,35,260,140,Fade(BLACK,0.75f));
                    string act_names[]={"No Action","Reroute","Throttle 10%","Throttle 20%"};
                    DrawText(("Reward:   "+to_string(last_result.reward)).c_str(),         W-265,40,17,WHITE);
                    DrawText(("Overloads:"+to_string(last_result.n_overloaded)).c_str(),    W-265,60,17,last_result.n_overloaded>0?RED:GREEN);
                    DrawText(("MaxUtil:  "+to_string((int)(last_result.max_severity*100))+"%").c_str(),W-265,80,17,last_result.max_severity>=0.9f?ORANGE:WHITE);
                    DrawText(("AvgUtil:  "+to_string((int)(last_result.avg_util*100))+"%").c_str(),  W-265,100,17,WHITE);
                    DrawText(("Shed:     "+to_string((int)last_result.power_shed)+" MW").c_str(),     W-265,120,17,last_result.power_shed>0?ORANGE:WHITE);
                    DrawText(("OBS dim:  "+to_string(obs_layout.total())).c_str(),          W-265,140,17,GRAY);
                }

                // Inspector
                if(sel_node){
                    DrawRectangle(10,140,270,130,Fade(BLACK,0.8f));
                    DrawText(("Type: "+sel_node->node_type).c_str(),15,145,16,YELLOW);
                    DrawText(("CSV Demand: "+to_string((int)sel_node->csv_demand)).c_str(),15,165,18,WHITE);
                    DrawText(("Curr Load:  "+to_string((int)sel_node->current_downstream_demand)).c_str(),15,185,18,WHITE);
                    DrawText(("Max Limit:  "+to_string((int)sel_node->max_limit)).c_str(),15,205,18,WHITE);
                    if(G.node_overloads_visual.count(sel_node)) DrawText("OVERLOADED",15,225,18,RED);
                } else if(sel_edge){
                    DrawRectangle(10,140,260,80,Fade(BLACK,0.8f));
                    DrawText(("Load: "+to_string((int)sel_edge->current_load)+"/"+
                              to_string((int)sel_edge->max_load)).c_str(),15,145,18,WHITE);
                    float u=sel_edge->current_load/max(sel_edge->max_load,0.01f);
                    DrawText(("Util: "+to_string((int)(u*100))+"%").c_str(),15,165,18,u>=0.9f?RED:u>=0.7f?ORANGE:GREEN);
                    DrawText(("Loss: "+to_string(sel_edge->loss)).c_str(),15,185,18,GRAY);
                }

                if(!rl_mode){
                    DrawRectangle(W-270,35,260,120,Fade(BLACK,0.7f));
                    DrawText("[I] Random Demand Surge",W-265,40,17,WHITE);
                    DrawText("[O] Fix Overloads",       W-265,60,17,WHITE);
                    DrawText("[U] Upgrade Node",         W-265,80,17,WHITE);
                    DrawText("[C] Clear Highlights",     W-265,100,17,WHITE);
                    DrawText("[R] RL Mode",              W-265,120,17,ORANGE);
                }

                EndDrawing();
            } break;
            }
        }

        delete bridge;
        CloseWindow();
        return 0;
    }
