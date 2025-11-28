import java.util.*;

// Daily Problem
public class DailySolutionOne {
    // 1262. Greatest Sum Divisible By Three
    public int maxSumDivThree(int[] nums) {
        int length = nums.length;
        int[] sum = new int[]{0,0,0};
        int[] tmpSum = new int[]{0,0,0};
        int remainder = 0;
        int index = 0;
        for(int i=0;i<length;i++){
            remainder = nums[i] % 3;
            for(int j=0;j<3;j++){
                index = (3+j-remainder) % 3;
                if(sum[index] % 3 != index) tmpSum[j] = sum[j];
                else tmpSum[j] = sum[index] + nums[i];
            }
            for (int j=0;j<3;j++) {
                sum[j] = Math.max(sum[j], tmpSum[j]);
            }
        }
        return sum[0];
    }
    // 1437. Check if All 1's Are at Least Length K Places Away
    public boolean kLengthApart(int[] nums, int k) {
        int count = 0;
        boolean firstTime = true;
        for(var e : nums){
            if(e == 1){
                if(firstTime) firstTime = false;
                else if(count < k) return false;
                count = 0;
            }
            else if(!firstTime){
                count += 1;
            }
        }
        return true;
    }
    // 3190. Find Minimum Operations to Make All Elements Divisible by Three
    public int minimumOperations(int[] nums) {
        int res = 0;
        for(var e : nums){
            if(e % 3 != 0) res += 1;
        }
        return res;
    }
    // 717. 1-bit and 2-bit Characters
    public boolean isOneBitCharacter(int[] bits) {
        int bits_length = bits.length;
        if(bits_length == 0) return false;
        if(bits_length == 1) return  bits[0] == 0;
        int index = 0;
        while (index < bits_length - 1){
            if(bits[index] == 0) index += 1;
            else index += 2;
        }
        if(index == bits_length - 1) return bits[index] == 0;
        return false;
    }
    // 1930. Unique Length-3 Palindromic Subsequences
    public int countPalindromicSubsequence(String s) {
        int s_length = s.length();
        if(s_length < 3) return 0;
        int[] first = new int[26];
        int[] last = new int[26];
        Arrays.fill(first, s_length);
        Arrays.fill(last, 0);
        int index = 0;
        for(int i=0;i<s_length;i++){
            index = s.charAt(i) - 'a';
            if(first[index] == s_length) first[index] = i;
            last[index] = i;
        }
        int ans = 0;
        for(int i=0;i<26;i++){
            if(first[i] < last[i]){
                ans += s.substring(first[i] + 1, last[i]).chars().distinct().count();
            }
        }
        return ans;
    }
    // 3217. Delete nodes from Linked List Present in Arrays
    public ListNode modifiedList(int[] nums, ListNode head) {
        var checkSet = new HashSet<>();
        for(var e : nums) checkSet.add(e);
        ListNode dummy = new ListNode(0, head);
        ListNode prev = dummy;
        ListNode cur = head;
        while (cur != null){
            if(checkSet.contains(cur.val))
                prev.next = cur.next;
            else
                prev = prev.next;
            cur = cur.next;
        }
        return dummy.next;
    }
    // 757. Set Intersection size at least Two
    public int intersectionSizeTwo(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[1] == o2[1]) return o2[0] - o1[0];
                return o1[1] - o2[1];
            }
        });

        int res = 0;
        int secondLastPoint = -1;
        int lastPoint = -1;
        int start = 0;
        int end = 0;

        for(var interval : intervals){
            start = interval[0];
            end = interval[1];

            if(start <= secondLastPoint) continue;
            if(start > lastPoint){
                res += 2;
                lastPoint = end;
                secondLastPoint = end-1;
            }
            else{
                res += 1;
                secondLastPoint = lastPoint;
                lastPoint = end;
            }
        }

        return res;
    }
    // 1018. Binary Prefix Divisible By 5
    public List<Boolean> prefixesDivBy5(int[] nums) {
        List<Boolean> answers = new ArrayList<>(nums.length);
        int x = 0;
        for(var e : nums){
            x = (x*2 + e)%5;
            answers.add(x==0);
        }
        return answers;
    }
    // 1015. Smallest Integer Divisible By K
    public int smallestRepunitDivByK(int k) {
        if(k == 1) return 1;
        if(k % 2 == 0) return -1;
        int resSlow = 1;
        int slow = 1;
        int resFast = 2;
        int fast = 11 % k;
        while (slow != fast){
            slow = (slow * 10 + 1) % k;
            fast = (fast * 10 + 1) % k;
            fast = (fast * 10 + 1) % k;
            resSlow += 1;
            resFast += 2;
            if(slow == 0) return resSlow;
            else if(fast == 0) return resFast;
        }
        return -1;
    }
    // 2435. Paths in Matrix Whose Sum is Divisible by K
    public int numberOfPaths(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        int module = (int) 1e9 + 7;
        int[][] dp = new int[m * n][k];
        for(int i=0;i<m*n;i++){
            for(int h=0;h<k;h++) {
                dp[i][h] = -1;
            }
        }
        return bt_numberOfPaths(0,0,0,k,grid,m,n,dp,module);
    }
    private int bt_numberOfPaths(int i, int j, int reminder, int k, int[][] grid, int m, int n, int[][] dp, int module){
        if((i == m-1) && (j == n-1)) {
            int newReminder = (reminder + grid[i][j]) % k;
            return newReminder == 0 ? 1 : 0;
        }
        int index = i * n + j;
        if(dp[index][reminder] != -1) return dp[index][reminder];
        int newReminder = (grid[i][j] + reminder) % k;
        int pathCount = 0;
        if(i == m-1){
            pathCount = bt_numberOfPaths(i, j+1, newReminder, k, grid, m, n, dp, module);
        }
        else if(j == n-1){
            pathCount = bt_numberOfPaths(i+1, j, newReminder, k, grid, m, n, dp, module);
        }
        else{
            pathCount = bt_numberOfPaths(i, j+1, newReminder, k, grid, m, n, dp, module)
                                + bt_numberOfPaths(i+1, j, newReminder, k, grid, m, n, dp, module);
        }
        dp[index][reminder] = pathCount % module;
        return dp[index][reminder];
    }
    // 3381. Maximum Subarray Sum With Length Divisible By K (DO 2/3)
    public long maxSubarraySum(int[] nums, int k) {
        long res = Long.MIN_VALUE;
        long prefix = 0;
        long[] minPrefix = new long[k];
        Arrays.fill(minPrefix, Long.MAX_VALUE / 2);
        minPrefix[k-1] = 0;
        for(int i=0;i<nums.length;i++){
            prefix += nums[i];
            res = Math.max(res, prefix - minPrefix[i%k]);
            minPrefix[i%k] = Math.min(minPrefix[i%k], prefix);
        }
        return res;
    }
    // 2872. Maximum Number of K-Divisible Components (DO 1/3)
    public int maxKDivisibleComponents(int n, int[][] edges, int[] values, int k) {
        List<Integer>[] graph = new List[n];
        for(int i=0;i<n;i++){
            graph[i] = new ArrayList<>();
        }
        for(var edge : edges){
            int firstNode = edge[0];
            int secondNode = edge[1];
            graph[firstNode].add(secondNode);
            graph[secondNode].add(firstNode);
        }
        int[] res = new int[1];
        dfs_MaxKDivisibleComponents(graph, 0, -1, values, k, res);
        return res[0];
    }
    private long dfs_MaxKDivisibleComponents(List<Integer>[] graph, int curNode, int parentNode, int[] value, int k, int[] res){
        long subTreeSum = value[curNode];
        for(int childNode : graph[curNode]){
            if(childNode != parentNode){
                subTreeSum += dfs_MaxKDivisibleComponents(graph, childNode, curNode, value, k, res);
            }
        }
        if(subTreeSum % k == 0) res[0] += 1;
        return subTreeSum;
    }
}
