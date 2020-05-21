/************************************************************************
	> File Name: implus.cpp
	> Author:  implus for speed up
	> Mail:    implusdream@gmail.com
	> Created Time: Tue 29 Mar 2016 05:14:58 AM PDT
 ************************************************************************/

#include<iostream>
#include<bits/stdc++.h>
#include<tr1/unordered_map>
using namespace std;
using namespace std::tr1;

vector<string> idx2word;
void idx2word2idx2word(char * idx2word_file){
    fstream f(idx2word_file, ios::in);
    cerr<<idx2word_file<<" idx2word load..."<<endl;
    string str;
    while(f >> str){
        idx2word.push_back(str);
    }
}

template<class Flow = int, class Cost = double>
struct MinCostFlow {
	struct Edge {
		int t;
		Flow f;
		Cost c;
		Edge*next, *rev;
		Edge(int _t, Flow _f, Cost _c, Edge*_next) :
				t(_t), f(_f), c(_c), next(_next) {
		}
	};

	vector<Edge*> E;
    void clearAll(){
        for(int i = 0; i < E.size(); i++)
            clear(E[i]);
        E.clear();
        word2pt.clear(), pos2pt.clear(), pt2pos.clear();
    }
    void clear(Edge* s){
        if(s == NULL) return;
        clear(s->next);
        delete s;
    }

	int addV() { E.push_back((Edge*) 0); return E.size() - 1; }

	Edge* makeEdge(int s, int t, Flow f, Cost c) {
		return E[s] = new Edge(t, f, c, E[s]);
	}

	void addEdge(int s, int t, Flow f, Cost c) {
		Edge*e1 = makeEdge(s, t, f, c), *e2 = makeEdge(t, s, 0, -c);
		e1->rev = e2, e2->rev = e1;
	}


	pair<Flow, Cost> minCostFlow(int vs, int vt) { //flow,cost
		int n = E.size();
		Flow flow = 0; Cost cost = 0;
		const Cost MAX_COST = numeric_limits<Cost>::max();
		const Flow MAX_FLOW = numeric_limits<Flow>::max();
		for (;;) {
			vector<Cost> dist(n, MAX_COST);
			vector<Flow> am(n, 0);
			vector<Edge*> prev(n);
			vector<bool> inQ(n, false);
			queue<int> que;

			dist[vs] = 0;
			am[vs] = MAX_FLOW;
			que.push(vs);
			inQ[vs] = true;

			while (!que.empty()) {
				int u = que.front();
				Cost c = dist[u];
				que.pop();
				inQ[u] = false;
				for (Edge*e = E[u]; e; e = e->next)
					if (e->f > 0) {
						Cost nc = c + e->c;
						if (nc < dist[e->t]) {
							dist[e->t] = nc;
							prev[e->t] = e;
							am[e->t] = min(am[u], e->f);
							if (!inQ[e->t]) {
								que.push(e->t);
								inQ[e->t] = true;
							}
						}
					}
			}

			if (dist[vt] == MAX_COST)
				break;

			Flow by = am[vt];
			int u = vt;
			flow += by;
			cost += by * dist[vt];
            //cerr<<"Flow = "<<flow<<" Cost = "<<cost<<" | ";
			while (u != vs) {
				Edge*e = prev[u];
				e->f -= by;
				e->rev->f += by;
				u = e->rev->t;
			}
		}

		return make_pair(flow, cost);
	}
    
    unordered_map<int, int> word2pt, pos2pt, pt2pos;
    int vs, vt;
    unordered_map<int, int> getMap(){
        unordered_map<int, int> res;
        for(unordered_map<int,int>::iterator it = word2pt.begin(); it != word2pt.end(); it++){
            int word = it->first, u = it->second;
            for(Edge* e = E[u]; e; e = e->next){
                if(e->f == 0){ // flow only this
                    int pos = pt2pos[e->t];
                    res[word] = pos;
                    break;
                }
            }
        }
        return res;
    }
};

MinCostFlow<int, double> MCF;

vector<vector<double> > vvdx, vvdy;
vector<vector<int> >  element_array;

const int MAX_LEN = 12345678;
char buf[MAX_LEN];

void file2vvd(char* filename, vector<vector<double> >& vvd){
    fstream f(filename, ios::in);
    while(f.getline(buf, MAX_LEN)){
        stringstream ss(buf); double val;
        vvd.push_back(vector<double>());
        while(ss >> val){
            vvd[vvd.size() - 1].push_back(val);
        }
    }
    f.close();
}
void element_array2file(char* filename){
    cerr<<filename<<" file to be saved"<<endl;
    fstream f(filename, ios::out);
    for(int i = 0; i < element_array.size(); i++){
        for(int j = 0; j < element_array[i].size(); j++){
            f << element_array[i][j] + 1 << " ";
        }
        f << "\n";
    }
    f.close();
}

void element_array2text(const char* filename){
    cerr<<filename<<" text string file to be saved"<<endl;
    fstream f(filename, ios::out);
    for(int i = 0; i < element_array.size(); i++){
        for(int j = 0; j < element_array[i].size(); j++){
            int v = element_array[i][j];
            if(v == -1){
                f << "<null>" <<" ";
            }else{
                f<< idx2word[v]<<" ";
            }
        }
        f <<"\n";
    }
    f.close();
}

void prt(string info){
    cerr<<"--------------------------- "<<info<<" ----------------------------------"<<endl;
}

int main(int argc, char* argv[]){
    ios_base::sync_with_stdio(false);
    // 1, 2, 3, 4
    idx2word2idx2word(argv[4]);
    cerr<<" read files into vvdx vvdy"<<endl;
    file2vvd(argv[1], vvdx);
    file2vvd(argv[2], vvdy);


    int vocab_size = vvdx.size();    // it must be the word vocabulary
    int vocab_sqrt = vvdx[0].size();
    cerr<<"vocab_size = "<<vocab_size<<endl;
    cerr<<"vocab_sqrt = "<<vocab_sqrt<<endl;
    
    // assign x
    prt("assign x by MCMF");
    cerr<<"clearAll and add points"<<endl;
    MCF.clearAll(); MCF.vs = MCF.addV(); MCF.vt = MCF.addV();
    for(int word = 0; word < vocab_size; word++){
        MCF.word2pt[word] = MCF.addV();
    }
    for(int x = 0; x < vocab_sqrt; x++){
        MCF.pos2pt[x]     = MCF.addV();
        MCF.pt2pos[MCF.pos2pt[x]] = x;
    }
    cerr<<"add edges by NLL loss from training set"<<endl;
	//void addEdge(int s, int t, Flow f, Cost c) {
    for(int word = 0; word < vocab_size; word++){
        MCF.addEdge(MCF.vs, MCF.word2pt[word], 1, 0.0);
    }
    for(int x = 0; x < vocab_sqrt; x++){
        MCF.addEdge(MCF.pos2pt[x], MCF.vt, vocab_sqrt, 0.0);
    }
    for(int word = 0; word < vocab_size; word++){
        for(int x = 0; x < vocab_sqrt; x++){
            // vvdx[word][x] is the NLL loss word->x
            MCF.addEdge(MCF.word2pt[word], MCF.pos2pt[x], 1, vvdx[word][x]);
        }
    }
    cerr<<"run MCMF for dim x adjust"<<endl;
	//pair<Flow, Cost> minCostFlow(int vs, int vt) { //flow,cost
    double st = clock();
    pair<int, double> pid = MCF.minCostFlow(MCF.vs, MCF.vt);
    cerr<<"max flow = "<<pid.first<<" min cost = "<<pid.second<<endl;
    cerr<<(clock() - st) / CLOCKS_PER_SEC / 60 <<" mins for MCMF dim x adjust"<<endl;


    for(int i = 0; i < vocab_sqrt; i++){
        //element_array.push_back(vector<int>(vocab_sqrt, -1));
        element_array.push_back(vector<int>());
    }
    unordered_map<int, int> uiix = MCF.getMap();
    for(unordered_map<int, int>::iterator it = uiix.begin(); it != uiix.end(); it++){
        int word = it->first, pos = it->second;
        assert(word < vocab_size); assert(pos < vocab_sqrt);
        element_array[pos].push_back(word);
    }
    cerr<<"assign words into dim x finished"<<endl;

    prt("assign words into dim y according to every x subset begin");
    for(int x = 0; x < vocab_sqrt; x++){ 
        // for each x to reassign y
        cerr<<"for x = "<<x<<" clearAll and add points"<<endl;
        MCF.clearAll(); MCF.vs = MCF.addV(); MCF.vt = MCF.addV();
        for(int i = 0; i < element_array[x].size(); i++){
            int word = element_array[x][i];
            MCF.word2pt[word] = MCF.addV();
        }
        for(int i = 0; i < vocab_sqrt; i++){
            MCF.pos2pt[i] = MCF.addV();
            MCF.pt2pos[MCF.pos2pt[i]] = i;
        }
        cerr<<"for x = "<<x<<" add edges!"<<endl;
        for(int i = 0; i < element_array[x].size(); i++){
            int word = element_array[x][i];
            MCF.addEdge(MCF.vs, MCF.word2pt[word], 1, 0.0);
        }
        for(int i = 0; i < vocab_sqrt; i++){
            MCF.addEdge(MCF.pos2pt[i], MCF.vt, 1, 0.0);
        }
        for(int i = 0; i < element_array[x].size(); i++){
            int word = element_array[x][i];
            for(int j = 0; j < vocab_sqrt; j++){
                MCF.addEdge(MCF.word2pt[word], MCF.pos2pt[j], 1, vvdy[word][j]);
            }
        }
        cerr<<"for x = "<<x<<" run MCMF for dim y adjust"<<endl;
        double st = clock();
        pair<int, double> pid = MCF.minCostFlow(MCF.vs, MCF.vt);
        cerr<<"max flow = "<<pid.first<<" min cost = "<<pid.second<<endl;
        cerr<<(clock() - st) / CLOCKS_PER_SEC / 60 <<" mins for MCMF dim y adjust"<<endl;

        // after network build, assign element_array[x] ->(vocab_sqrt, -1)
        element_array[x].assign(vocab_sqrt, -1);

        unordered_map<int, int> uiiy = MCF.getMap();
        for(unordered_map<int, int>::iterator it = uiiy.begin(); it != uiiy.end(); it++){
            int word = it->first, pos = it->second;
            assert(word < vocab_size); assert(pos < vocab_sqrt);
            element_array[x][pos] = word;
        }
    }

    element_array2file(argv[3]);
    string tmpfile = argv[3]; tmpfile += ".dic_string";tmpfile += argv[5];
    element_array2text(tmpfile.c_str());
    return 0;
}
