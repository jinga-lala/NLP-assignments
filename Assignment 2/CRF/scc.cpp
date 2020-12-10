#include <bits/stdc++.h>
using namespace std;

void topo(stack<int> &st, vector<int> *v, bool* vs, int rt)
{
	vs[rt]=1;
	for(auto it:v[rt])
	{
		if(!vs[it])
		{
			topo(st, v, vs, rt);
		}
	}
	st.push(rt);
}

void find_rch(vector<int>* rch, vector<int>* mp, int rt, bool* vs)
{
	vs[rt]=1;
	vector<int> st;
	for(auto it:v[rt])
	{
		if(!vs[it])
		{
			find_rch(rch, mp, it, vs);
		}
		for(auto ii:rch[it])
		{
			st.insert(ii);
		}
	}
	rch[rt]=st;
}

int main()
{
	int n;
	cin>>n;
	map<int, string> id;
	for(int i=0;i<n;i++)
	{
		cout<<"Plan";
		string g;cin>>g;
		id[g]=i;
	}
	int m;
	cin>>m;
	vector<int> v[n];
	for(int i=0;i<m;i++)
	{
		string a,b;
		cin>>a>>b;
		v[id[a]].push_back(id[b]);
	}
	vector<int> rch[n];
	bool vs[n];
	fill(vs, vs+n, 0);
	find_rch(rch, v, 0, vs);
	bool rh[n][n];
	for(int i=0;i<n;i++)
	{
		fill(rh[i], rh[i]+n, 0);
		for(auto it:rch[i])
		{
			rh[i][it]=1;
		}
	}
	for(int i=0;i<n;i++)
		for(int j=i+1;j<n;j++)
			if(!rh[j][i])
				v[i].push_back(j);

	fill(vs, vs+n, 0);
	stack<int> st;
	topo(st, v, vs, 0);
	for(auto it:st)
		cout<<it<<" ";
	return 0;
}