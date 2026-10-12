#include<bits/stdc++.h>
using namespace std;

int main(){
   int n;
   cin>>n;
   unordered_map<int,int> mp;
   vector<int> v(n);
   for(int i=0; i<n; i++){
      cin>>v[i];
      mp[v[i]] = i;
   }
   int start = mp[1], ans = 1;

   for(int i=2; i<=n; i++){
      if(mp[i]>=start){
         start = mp[i];
      }else{
         start = mp[i];
         ans++;
      }
   }

   cout<<ans<<endl;
}