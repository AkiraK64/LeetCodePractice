import java.util.*;

/** Daily Problem */
public class DailySolution {
    // 2154. Keep Multiplying Found Values By Two
    public int findFinalValue(int[] nums, int original) {
        HashSet<Integer> numSet = new HashSet<>();
        for(var num : nums) numSet.add(num);
        int res = original;
        while (numSet.contains(res))
            res *= 2;
        return res;
    }
    // 1262. Greatest Sum Divisible By Three
    /** DP Bottom-Up */
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
    /** Array & Hashing */
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
    /** Array & Hashing */
    public int minimumOperations(int[] nums) {
        int res = 0;
        for(var e : nums){
            if(e % 3 != 0) res += 1;
        }
        return res;
    }
    // 717. 1-bit and 2-bit Characters
    /** Array & Hashing */
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
    /** Array & Hashing */
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
                ans += (int) s.substring(first[i] + 1, last[i]).chars().distinct().count();
            }
        }
        return ans;
    }
    // 3217. Delete nodes from Linked List Present in Arrays
    /** Linked List */
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
    /** Greedy */
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
    /** Array & Hashing */
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
    /** Two Pointers */
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
    /** 3D-DP Top-Down */
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
    // 3381. Maximum Subarray Sum With Length Divisible By K
    /** Array & Hashing */
    public long maxSubarraySum(int[] nums, int k) {
        long prefix = 0;
        long res = Long.MIN_VALUE;
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
    // 2872. Maximum Number of K-Divisible Components (DO 2/3)
    /** DFS */
    public int maxKDivisibleComponents(int n, int[][] edges, int[] values, int k) {
        List<Integer>[] graph = new List[n];
        for(int i=0;i<n;i++)
            graph[i] = new ArrayList<>();
        int first = 0;
        int second = 0;
        for(var edge : edges){
            first = edge[0];
            second = edge[1];
            graph[first].add(second);
            graph[second].add(first);
        }
        int[] res = new int[1];
        dfs_MaxKDivisibleComponents(graph, 0, -1, values, k, res);
        return res[0];
    }
    private long dfs_MaxKDivisibleComponents(List<Integer>[] graph, int cur, int par, int[] values, int k, int[] res){
        long subTreeSum = values[cur];
        for(var child : graph[cur]){
            if(child != par){
                subTreeSum += dfs_MaxKDivisibleComponents(graph, child, cur, values, k, res);
            }
        }
        if(subTreeSum % k == 0)
            res[0] += 1;
        return subTreeSum;
    }
    // 3512. Minimum Operations to Make Array Sum Divisible by K
    /** Array & Hashing */
    public int minOperations(int[] nums, int k) {
        int sum = 0;
        for(var num : nums){
            sum = (sum + num) % k;
        }
        return sum;
    }
    // 1590. Make Sum Divisible By P
    /** Array & Hashing */
    public int minSubarray(int[] nums, int p) {
        HashMap<Integer, Integer> prefix = new HashMap<>();
        prefix.put(0, -1);
        int rTotal = 0;
        for (int num : nums) rTotal = (rTotal + num) % p;
        if(rTotal == 0) return 0;
        int reminder = 0;
        int index = 0;
        int nReminder = 0;
        int res = nums.length;
        for(int i=0;i<nums.length;i++){
            reminder = (reminder + nums[i]) % p;
            nReminder = (((reminder - rTotal) % p) + p) % p;
            if(prefix.containsKey(nReminder)){
                index = prefix.get(nReminder);
                res = Math.min(res, (i - index));
            }
            prefix.put(reminder, i);
        }
        return res == nums.length ? -1 : res;
    }
    // 2141. Maximum Running Time of N Computers
    /**  Binary Search */
    public long maxRunTime(int n, int[] batteries) {
        long l = 0;
        long r = 0;
        for(var bt : batteries) r += bt;
        long totalUsableCapacity = 0;
        long mid = 0;
        long res = 0;
        while (l <= r){
            mid = (l + r) / 2;
            totalUsableCapacity = 0;
            for(var bt : batteries) totalUsableCapacity += Math.min(bt, mid);
            if(totalUsableCapacity >= n * mid){
                l = mid + 1;
                res = Math.max(res, mid);
            }
            else r = mid - 1;
        }
        return res;
    }
    // 3623. Count Number Of Trapezoids I
    /** Array & Hashing */
    public int countTrapezoids(int[][] points) {
        int MOD = 1000000007;
        HashMap<Integer, Integer> map = new HashMap<>();
        for(var point : points){
            map.put(point[1], map.getOrDefault(point[1], 0) + 1);
        }
        long newValue = 0;
        long totalSum = 0;
        long totalSumOfSquares = 0;
        for (var value : map.values()){
            if(value == 1) continue;
            newValue = ((long) value * (value - 1)) / 2;
            newValue %= MOD;
            totalSum = (totalSum + newValue) % MOD;
            totalSumOfSquares = (totalSumOfSquares + (newValue * newValue) % MOD) % MOD;
        }
        long total = 0;
        total = (totalSum * totalSum) % MOD;
        total = (((total - totalSumOfSquares) % MOD) + MOD) % MOD;
        if(total % 2 == 0) total = total / 2;
        else total = (total + MOD) / 2;
        return (int)total;
    }
    // 3625. Count Number Of Trapezoids II
    /** Array & Hashing */
    public int countTrapezoidsII(int[][] points) {
        Map<Double, Map<Double, Integer>> parallels = new HashMap<>();
        Map<Integer, Map<Double, Integer>> sameMidPoints = new HashMap<>();
        double a, b;
        int x1, x2, y1, y2, dx, dy, midPoint;
        int n = points.length;
        for(int i=0;i<n;i++){
            x1 = points[i][0];
            y1 = points[i][1];
            for(int j=i+1;j<n;j++){
                x2 = points[j][0];
                y2 = points[j][1];
                dx = x2 - x1;
                dy = y2 - y1;
                a = (dx == 0 ? Double.MAX_VALUE : 1.0 * dy / dx);
                b = (dx == 0 ? x1 : 1.0 * (y1 * dx - x1 * dy) / dx);
                if(a == -0.0) a = 0.0;
                if(b == -0.0) b = 0.0;
                if(parallels.containsKey(a)){
                    var subMap = parallels.get(a);
                    subMap.put(b, subMap.getOrDefault(b, 0) + 1);
                }
                else{
                    Map<Double, Integer> subMap = new HashMap<>();
                    subMap.put(b, 1);
                    parallels.put(a, subMap);
                }
                midPoint = (x1 + x2 + 2000) * 4000 + (y1 + y2 + 2000);
                if(sameMidPoints.containsKey(midPoint)){
                    var subMap = sameMidPoints.get(midPoint);
                    subMap.put(a, subMap.getOrDefault(a, 0) + 1);
                }
                else{
                    Map<Double, Integer> subMap = new HashMap<>();
                    subMap.put(a, 1);
                    sameMidPoints.put(midPoint, subMap);
                }
            }
        }
        int res = 0;
        for(var e : parallels.values()){
            int s = 0;
            for(var t : e.values()){
                res += s * t;
                s += t;
            }
        }
        for (var e : sameMidPoints.values()){
            int s = 0;
            for(var t : e.values()){
                res -= s * t;
                s += t;
            }
        }
        return res;
    }
    // 2211. Count Collisions on a Road
    /** Iterator */
    public int countCollisions(String directions) {
        int res = 0;
        int n = directions.length();
        int l = 0;
        int r = n-1;
        while (l < n && directions.charAt(l) == 'L')
            l += 1;
        while (r >= 0 && directions.charAt(r) == 'R')
            r -= 1;
        while (l <= r){
            if(directions.charAt(l) != 'S') res += 1;
            l += 1;
        }
        return res;
    }
    // 3432. Count Partitions with Even Sum Difference
    /** Array & Hashing */
    public int countPartitions(int[] nums) {
        int sum = 0;
        for(var num : nums) sum += num;
        return sum % 2 == 0 ? nums.length - 1 : 0;
    }
    // 3578. Count Partitions With Min-Max Difference At most K
    /** Two Pointers */
    public int countPartitions(int[] nums, int k) {
        int MOD = (int)(1e9 + 7);
        int n = nums.length;
        int l = 1;
        int[] res = new int[n+1];
        int[] prefix = new int[n+1];
        res[0] = 1;
        prefix[0] = 1;
        TreeMap<Integer, Integer> numMap = new TreeMap<>();
        for(int r=1;r<=n;r++){
            int cur = nums[r-1];
            numMap.merge(cur, 1, Integer::sum);
            while (numMap.lastKey() - numMap.firstKey() > k){
                if(numMap.merge(nums[l-1], -1, Integer::sum) == 0)
                    numMap.remove(nums[l-1]);
                l += 1;
            }
            res[r] = (prefix[r-1] - (l >= 2 ? prefix[l-2] : 0)) % MOD;
            res[r] = (res[r] + MOD) % MOD;
            prefix[r] = (prefix[r-1] + res[r]) % MOD;
        }
        return res[n];
    }
    // 1523. Count Odd Numbers in an Interval Range
    /** Math */
    public int countOdds(int low, int high) {
        if(high % 2 == 0) high -= 1;
        if(low % 2 == 0) low += 1;
        return (high - low) / 2 + 1;
    }
    // 1925. Count Squares Sum Triples
    /** Math */
    public int countTriples(int n) {
        int m = (int)Math.sqrt(n * n * 0.5);
        int sum = 0;
        int sqrt = 0;
        int res = 0;
        for(int i=1;i<=m;i++){
            for(int j=i+1;j<n;j++){
                sum = i * i + j * j;
                if(sum > n * n) break;
                sqrt = (int)Math.sqrt(sum);
                if(sum == sqrt * sqrt) res += 2;
            }
        }
        return res;
    }
    // 3583. Count Special Triplets
    /** Array & Hashtable */
    public int specialTriplets(int[] nums){
        int MOD = (int)(1e9+7);
        HashMap<Integer, Integer> rCount = new HashMap<>();
        HashMap<Integer, Integer> lCount = new HashMap<>();
        long res = 0;
        for(var num : nums) rCount.merge(num, 1, Integer::sum);
        for(var num : nums){
            rCount.merge(num, -1, Integer::sum);
            int target = num * 2;
            long leftTargetCount = rCount.getOrDefault(target, 0);
            long rightTargetCount = lCount.getOrDefault(target, 0);
            res = (res + (leftTargetCount * rightTargetCount) % MOD) % MOD;
            lCount.merge(num, 1, Integer::sum);
        }
        return (int)res;
    }
    // 3577. Count the Number Of Computer Unlocking Permutations
    /** Math */
    public int countPermutations(int[] complexity) {
        int MOD = (int)(1e9+7);
        long res = 1;
        int length = complexity.length;
        if(length == 1) return 1;
        for(int i=1;i<length;i++)
            if(complexity[i] <= complexity[0]) return 0;
        length -= 1;
        while (length > 0){
            res *= length;
            res %= MOD;
            length -= 1;
        }
        return (int)res;
    }
    // 3531. Count Covered Building
    /** Array & Hashing */
    public int countCoveredBuildings(int n, int[][] buildings) {
        int[] north = new int[n+1];
        int[] south = new int[n+1];
        int[] west = new int[n+1];
        int[] east = new int[n+1];
        Arrays.fill(south, n+1);
        Arrays.fill(west, n+1);
        for(int[] building : buildings){
            int x = building[0];
            int y = building[1];
            if(y > north[x]) north[x] = y;
            if(y < south[x]) south[x] = y;
            if(x > east[y]) east[y] = x;
            if(x < west[y]) west[y] = x;
        }
        int res = 0;
        for (int[] building : buildings){
            int x = building[0];
            int y = building[1];
            if(x < east[y] && x > west[y] && y < north[x] && y > south[x])
                res += 1;
        }
        return res;
    }
    // 3433. Count Mentions Per User
    /** Array & Hashing */
    public int[] countMentions(int numberOfUsers, List<List<String>> events) {
        int[] mentions = new int[numberOfUsers];
        int[] startOnlineTime = new int[numberOfUsers];
        HashMap<String, Integer> compareString = new HashMap<>();
        compareString.put("OFFLINE", 0);
        compareString.put("MESSAGE", 1);
        events.sort(new Comparator<List<String>>() {
            @Override
            public int compare(List<String> o1, List<String> o2) {
                int timeStamp1 = Integer.parseInt(o1.get(1));
                int timeStamp2 = Integer.parseInt(o2.get(1));
                if(timeStamp1 == timeStamp2)
                    return compareString.get(o1.getFirst()) - compareString.get(o2.getFirst());
                return timeStamp1 - timeStamp2;
            }
        });
        for(var event : events){
            var title = event.getFirst();
            int timeStamp = Integer.parseInt(event.get(1));
            if(Objects.equals(title, "MESSAGE")){
                var mentions_string = event.get(2);
                if(Objects.equals(mentions_string, "ALL")){
                    for(int i=0;i<numberOfUsers;i++){
                        mentions[i] += 1;
                    }
                }
                else if(Objects.equals(mentions_string, "HERE")){
                    for(int i=0;i<numberOfUsers;i++){
                        if(startOnlineTime[i] <= timeStamp)
                            mentions[i] += 1;
                    }
                }
                else{
                    var ids = mentions_string.split(" ");
                    for(var id : ids){
                        int int_id = Integer.parseInt(id.substring(2));
                        mentions[int_id] += 1;
                    }
                }
            }
            else{
                int id = Integer.parseInt(event.get(2));
                startOnlineTime[id] = timeStamp + 60;
            }
        }
        return mentions;
    }
    // 3606. Coupon Code Validator
    /** Array & Hashing */
    public List<String> validateCoupons(String[] code, String[] businessLine, boolean[] isActive) {
        int n = code.length;
        List<String[]> list = new ArrayList<>(n);
        Map<String, Integer> categories = new HashMap<>();
        categories.put("electronics", 0);
        categories.put("grocery", 1);
        categories.put("pharmacy", 2);
        categories.put("restaurant", 3);
        for (int i=0;i<n;i++){
            if(!isActive[i]) continue;
            if(!categories.containsKey(businessLine[i])) continue;
            if(code[i].matches("^[a-zA-Z0-9_]+$")){
                list.add(new String[]{code[i], businessLine[i]});
            }
        }
        list.sort(new Comparator<String[]>() {
            @Override
            public int compare(String[] o1, String[] o2) {
                int i1 = categories.get(o1[1]);
                int i2 = categories.get(o2[1]);
                if(i1 == i2) return o1[0].compareTo(o2[0]);
                return i1 - i2;
            }
        });
        List<String> res = new ArrayList<>(list.size());
        for (var e : list) res.add(e[0]);
        return res;
    }
    // 2147. Number of Ways to Divide a Long Corridor
    /** Math */
    public int numberOfWays(String corridor) {
        int MOD = (int)(1e9 + 7);
        int n = corridor.length();
        int countS = 0;
        int countP = 0;
        long res = 1;
        for(int i=0;i<n;i++){
            var c = corridor.charAt(i);
            if(c == 'S'){
                if(countS == 2){
                    res = (res * (countP + 1)) % MOD;
                    countP = 0;
                    countS = 0;
                }
                countS += 1;
            }
            else if(countS == 2){
                countP += 1;
            }
        }
        return countS == 2 ? (int)res : 0;
    }
    // 2110. Number of Smooth Descent Periods of a Stock
    /** Two Pointers */
    public long getDescentPeriods(int[] prices) {
        int l = 0;
        int n = prices.length;
        long res = 0;
        for(int r=0;r<n;r++){
            if(r+1 < n && prices[r+1] + 1 == prices[r]){
                continue;
            }
            else{
                int m = r - l + 1;
                res += ((long) m * (m + 1)) / 2;
                l = r + 1;
            }
        }
        return res;
    }
}
