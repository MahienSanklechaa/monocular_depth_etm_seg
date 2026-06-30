#include <bits/stdc++.h>
#define int long long
#define BOOST ios::sync_with_stdio(false); cin.tie(nullptr);

using namespace std;

bool canVisitAll(const vector<vector<int>>& g) {
    int n = g.size();
    if (n == 0) return true;

    for (int start = 0; start < n; start++) {
        vector<int> vis(n, 0);
        queue<int> q;
        q.push(start);
        vis[start] = 1;
        int cnt = 1;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : g[u]) {
                if (!vis[v]) {
                    vis[v] = 1;
                    q.push(v);
                    cnt++;
                }
            }
        }

        if (cnt != n) return false;
    }
    return true;
}
inline void solve() {
    int n;
    cin >> n;

    vector<string> s(n);
    for (int i = 0; i < n; i++) {
        cin >> s[i];
    }

    vector<pair<int,int>>vp;

    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            if(s[i][j] == '#'){
                vp.push_back({i,j});
            }
        }
    }

    int ss=vp.size();
    vector<vector<int>>g(ss);

    for(int i=0; i<ss; i++) {
        int curr_x=vp[i].first, curr_y=vp[i].second;
        for(int j=0; j<ss; j++){
            if(i!=j) {
                int i_x=vp[j].first, i_y=vp[j].second;
                int dif_x=abs(i_x-curr_x);
                int dif_y=abs(i_y-curr_y);
                if(abs(dif_x-dif_y) <= 1) {
                    g[i].push_back(j);
                    g[j].push_back(i);
                }
            }
        }
    }

    for(auto it:vp) cout << it.first << " "  << it.second << '\n';
    for(auto it:g) {{for(auto el:it) cout << el << " ";} cout << '\n';}


   if (canVisitAll(g)) cout << "YES\n";
    else cout << "NO\n";

}

signed main() {
    BOOST
    int T;
    cin >> T;
    while (T--) solve();
    return 0;
}
