class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        l=len(connections)
        if l<n-2:
            return -1
        if l>n-1:
            return 0
        groups=[connections[1]]
        for i in range(1,l):
            m,n=connections[i]
            flag=0
            for k in range(len(groups)):
                if (m in groups(k)) or (n in groups(k)):
                    groups[k].append(n)
                    groups[k].append(m)
                    flag=1
            if flag==0:
                groups.append([m,n])
        return len(groups)-1

makeConnected(n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]])