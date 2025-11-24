import java.util.*;

/** <b>Neetcode 150</b> */
public class FirstSolution {
    /** <i>Array And Hash</i> */
    // 217. Contain Duplicate
    public boolean containsDuplicate(int[] nums) {
        var checkSet = new HashSet<Integer>();
        int capacity = nums.length;
        for(int i=0;i<capacity;i++){
            if(checkSet.contains(nums[i])) return true;
            checkSet.add(nums[i]);
        }
        return false;
    }
    // 1. Two Sum
    public int[] twoSum(int[] nums, int target) {
        int capacity = nums.length;
        int lastIndex = -1;
        var checkSet = new HashSet<Integer>();

        for(int i=0;i<capacity;i++){
            if(checkSet.contains(target - nums[i])){
                lastIndex = i;
                break;
            }
            checkSet.add(nums[i]);
        }

        for(int j=0;j<lastIndex;j++){
            if(nums[j] + nums[lastIndex] == target) return new int[]{j, lastIndex};
        }

        return new int[0];
    }
    // 242. Valid Anagram
    public boolean isAnagram(String s, String t) {
        int t_Size = t.length();
        int s_Size = s.length();
        if(t_Size != s_Size) return false;
        var checkMap = new HashMap<Character, Integer>();
        for(var c : s.toCharArray()){
            if(checkMap.containsKey(c)) checkMap.replace(c, checkMap.get(c)+1);
            else checkMap.put(c, 1);
        }
        for(var c : t.toCharArray()){
            if(checkMap.containsKey(c)){
                if(checkMap.get(c) == 0) return false;
                else checkMap.replace(c, checkMap.get(c)-1);
            }
            else return false;
        }
        return true;
    }
    // 49. Group Anagram
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, ArrayList<String>> resMap = new HashMap<>();
        int[] fArr = new int[26];
        StringBuilder key;
        String keyStr;
        for(var s : strs){
            for(var c : s.toCharArray()){
                fArr[c - 'a'] += 1;
            }
            key = new StringBuilder();
            for(int i=0;i<26;i++){
                if(fArr[i] != 0){
                    key.append(i + 'a');
                    key.append(fArr[i]);
                    fArr[i] = 0;
                }
            }
            keyStr = key.toString();
            if(resMap.containsKey(keyStr)) resMap.get(keyStr).add(s);
            else resMap.put(keyStr, new ArrayList<>(List.of(s)));
        }
        return new ArrayList<>(resMap.values());
    }


    /** <i>Two Pointers</i> */
    // 125. Valid Palindrome
    public boolean isPalindrome(String s) {
        int s_length = s.length();
        if(s_length == 0) return true;
        int left = 0;
        int right = s_length-1;
        var leftChar = s.charAt(left);
        var rightChar = s.charAt(right);
        while (left < right){
            leftChar = s.charAt(left);
            if(leftChar < 48 || (leftChar > 57 && leftChar < 65) || (leftChar > 90 && leftChar < 97) || leftChar > 122){
                left += 1;
                continue;
            }
            rightChar = s.charAt(right);
            if(rightChar < 48 || (rightChar > 57 && rightChar < 65) || (rightChar > 90 && rightChar < 97) || rightChar > 122){
                right -= 1;
                continue;
            }
            if(leftChar < 65){
                if(leftChar == rightChar){
                    left += 1;
                    right -= 1;
                }
                else return false;
            }
            else if(rightChar < 65){
                return false;
            }
            else{
                if(rightChar == leftChar || Math.abs(rightChar - leftChar) == 32){
                    left += 1;
                    right -= 1;
                }
                else return false;
            }
        }
        return true;
    }


    /** <i>Sliding Window</i> */
    // 567. Permutation in String
    public boolean checkInclusion(String s1, String s2) {
        int s1_length = s1.length();
        int s2_length = s2.length();
        if(s1_length > s2_length) return false;
        int[] s1_frequency = new int[26];
        int[] s2_frequency = new int[26];
        for(int i=0;i<s1_length;i++){
            s1_frequency[s1.charAt(i) - 'a'] += 1;
            s2_frequency[s2.charAt(i) - 'a'] += 1;
        }
        int matches = 0;
        for(int i=0;i<26;i++){
            if(s1_frequency[i] == s2_frequency[i])
                matches += 1;
        }
        int left = 0;
        for(int right=s1_length;right<s2_length;right++){
            if(matches == 26) return true;

            int index = s2.charAt(right) - 'a';
            s2_frequency[index] += 1;
            if(s2_frequency[index] == s1_frequency[index]) matches += 1;
            else if(s2_frequency[index] == (s1_frequency[index] + 1)) matches -= 1;

            index = s2.charAt(left) - 'a';
            s2_frequency[index] -= 1;
            if(s2_frequency[index] == s1_frequency[index]) matches += 1;
            else if(s2_frequency[index] == (s1_frequency[index] - 1)) matches -= 1;

            left += 1;
        }
        return matches == 26;
    }
    // 3. Longest Substring Without Repeating Characters
    public int lengthOfLongestSubstring(String s) {
        int sLength = s.length();
        if(sLength == 0) return 0;
        int l=0;
        int res=0;
        int curLength = 0;
        var checkSet = new HashSet<Character>();
        var curChar = s.charAt(0);
        var lChar = s.charAt(l);
        for(int i=0;i<sLength;i++){
            curChar = s.charAt(i);
            while (checkSet.contains(curChar)){
                lChar = s.charAt(l);
                checkSet.remove(lChar);
                l++;
            }
            checkSet.add(curChar);
            curLength = i-l+1;
            if(curLength > res) res = curLength;
        }
        return res;
    }
    // 121. Best Time to Buy and Sell Stock
    public int maxProfit(int[] prices) {
        int days = prices.length;
        int left = 0;
        int right = 1;
        int profit = 0;
        int expectedProfit = 0;
        while (right < days){
            expectedProfit = prices[right] - prices[left];
            if(expectedProfit > 0){
                right += 1;
                if(expectedProfit > profit) profit = expectedProfit;
            }
            else{
                left = right;
                right = left + 1;
            }
        }
        return profit;
    }
    // 424. Longest Repeating Character Replacement (DO 2/3)
    public int characterReplacement(String s, int k) {
        HashMap<Character, Integer> map = new HashMap<>();
        int maxFrequencyOfChar = 0;
        int s_length = s.length();
        int left = 0;
        int res = 0;
        for(int right=0;right<s_length;right++){
            Character charIndex = s.charAt(right);
            map.put(charIndex, map.getOrDefault(charIndex, 0) + 1);
            maxFrequencyOfChar = Math.max(maxFrequencyOfChar, map.get(charIndex));
            while ((right - left + 1) - maxFrequencyOfChar > k){
                charIndex = s.charAt(left);
                map.put(charIndex, map.get(charIndex) - 1);
                maxFrequencyOfChar = Math.max(maxFrequencyOfChar, map.get(charIndex));
                left += 1;
            }
            res = Math.max(res, right - left + 1);
        }
        return res;
    }


    /** <i>Stack</i> */
    // 20. Valid Parentheses
    public boolean isValid(String s) {
        Stack<Character> checkStack = new Stack<>();
        for(var c : s.toCharArray()){
            if(c == '(' || c == '{' || c == '[')
                checkStack.add(c);
            else{
                if(checkStack.isEmpty()) return false;
                var checkChar = checkStack.pop();
                if(checkChar == '(' && c != ')') return false;
                if(checkChar == '{' && c != '}') return false;
                if(checkChar == '[' && c != ']') return false;
            }
        }
        return checkStack.isEmpty();
    }


    /** <i>Binary Search</i> */
    // 704. Binary Search
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int middle = 0;
        while (left <= right){
            middle = (left + right) / 2;
            if(nums[middle] == target) return middle;
            else if(nums[middle] > target) right = middle - 1;
            else left = middle + 1;
        }
        return -1;
    }


    /** <i>Linked List</i> */
    // 206. Reverse Linked List
    public ListNode reverseList(ListNode head) {
        if(head == null) return null;
        ListNode prevNode = null;
        ListNode curNode = head;
        ListNode nextNode = head;
        while (curNode != null){
            nextNode = curNode.next;
            curNode.next = prevNode;
            prevNode = curNode;
            curNode = nextNode;
        }
        return prevNode;
    }
    // 21. Merge Two Sorted List
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0, null);
        ListNode cur = dummy;
        while (list1 != null && list2 != null){
            if(list1.val > list2.val){
                cur.next = list2;
                list2 = list2.next;
            }
            else{
                cur.next = list1;
                list1 = list1.next;
            }
            cur = cur.next;
        }

        while (list1 != null){
            cur.next = list1;
            list1 = list1.next;
            cur = cur.next;
        }

        while (list2 != null){
            cur.next = list2;
            list2 = list2.next;
            cur = cur.next;
        }

        cur.next = null;
        return dummy.next;
    }
    // 141. Linked List Cycle (Floyd's Algorithm) (DO 2/3)
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) return true;
        }
        return false;
    }


    /** <i>Tree</i> */
    // 226. Invert Binary Tree
    public TreeNode invertTree(TreeNode root) {
        if(root == null) return null;
        var leftRoot = invertTree(root.left);
        var rightRoot = invertTree(root.right);
        root.right = leftRoot;
        root.left = rightRoot;
        return root;
    }
    // 543. Diameter of Binary Tree
    public int diameterOfBinaryTree(TreeNode root) {
        int[] res = new int[1];
        dfs_DiameterOfBinaryTree(root, res);
        return res[0];
    }
    private int dfs_DiameterOfBinaryTree(TreeNode root, int[] res) {
        if (root == null) {
            return 0;
        }
        int left = dfs_DiameterOfBinaryTree(root.left, res);
        int right = dfs_DiameterOfBinaryTree(root.right, res);
        res[0] = Math.max(res[0], left + right);
        return 1 + Math.max(left, right);
    }
    // 104. Maximum Depth of Binary Tree
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return 1 + Math.max(leftDepth, rightDepth);
    }
    // 110. Balanced Binary Tree
    public boolean isBalanced(TreeNode root) {
        return dfs_IsBalanced(root)[0] == 1;
    }
    private int[] dfs_IsBalanced(TreeNode root){
        if(root == null) return new int[]{1,0};
        int[] left = dfs_IsBalanced(root.left);
        int[] right = dfs_IsBalanced(root.right);
        int isBalanced = 1;
        if(left[0] == 0 || right[0] == 0) isBalanced = 0;
        else if(Math.abs(left[1] - right[1]) > 1) isBalanced = 0;
        int heigth = 1 + Math.max(left[1], right[1]);
        return new int[]{isBalanced, heigth};
    }
    // 100. Same Tree
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p != null && q != null && p.val == q.val)
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        return false;
    }
    // 572. Subtree Of Another Tree
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if(subRoot == null) return true;
        if(root == null) return false;
        if(isSameTree(root, subRoot)) return true;
        return isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    /** <i>Heap & Priority Queue</i> */
    // 703. Kth Largest Element in a Stream (DO 1/3)
    public int kthLargestAdd(int k, int[] nums, int val){
        var kthIns = new KthLargest(k, nums);
        return kthIns.add(val);
    }
    // 1046 Last Stone Weight
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
        for(var e : stones) maxHeap.add(e);
        int first = 0;
        int second = 0;
        while (maxHeap.size() > 1){
            first = maxHeap.poll();
            if(!maxHeap.isEmpty()) {
                second = maxHeap.poll();
                maxHeap.add(first - second);
            }
            else break;
        }
        return maxHeap.isEmpty() ? 0 : maxHeap.peek();
    }

    /** <i>2-D Dynamic Programing</i> */
    // 494. Target Sum
    public int findTargetSumWays(int[] nums, int target) {
        int totalSum = 0;
        for(var e : nums) totalSum += e;
        int nums_length = nums.length;
        int tmpTotalSum = 2*totalSum+1;
        int[][] dp = new int[nums_length][tmpTotalSum];
        for(int i=0;i<nums_length;i++){
            for (int j=0;j<tmpTotalSum;j++){
                dp[i][j] = Integer.MIN_VALUE;
            }
        }
        return bt_findTargetSumWays(0, 0, nums, target, totalSum, dp);
    }
    private int bt_findTargetSumWays(int index, int total, int[] nums, int target, int totalSum, int[][] dp){
        if(index == nums.length) return total == target ? 1 : 0;
        if(dp[index][total+totalSum] != Integer.MIN_VALUE) return dp[index][total+totalSum];
        dp[index][total+totalSum] = bt_findTargetSumWays(index+1, total+nums[index], nums, target, totalSum, dp)
                + bt_findTargetSumWays(index+1, total-nums[index], nums, target, totalSum, dp);
        return dp[index][total+totalSum];
    }
    // 309. Best Time to Buy and Sell Stock with Cooldown
    public int maxProfitWithCooldown(int[] prices) {
        int days = prices.length;
        int[][] dp = new int[days][2];
        for(int i=0;i<days;i++){
            dp[i][0] = -1;
            dp[i][1] = -1;
        }
        return bt_maxProfitWithCooldown(0, 0, prices, dp);
    }
    private int bt_maxProfitWithCooldown(int i, int state, int[] prices, int[][] dp){
        if(i == prices.length) return 0;
        if(state == 3) return bt_maxProfitWithCooldown(i+1, 0, prices, dp);
        if(dp[i][state] != -1) return dp[i][state];
        int profit = bt_maxProfitWithCooldown(i+1, state, prices, dp);
        int expectedProfit = 0;
        if(state == 0){
            expectedProfit = -prices[i] + bt_maxProfitWithCooldown(i+1, 1, prices, dp);
            dp[i][state] = Math.max(expectedProfit, profit);
        }
        else{
            expectedProfit = prices[i] + bt_maxProfitWithCooldown(i+1,3, prices, dp);
            dp[i][state] = Math.max(expectedProfit, profit);
        }
        return dp[i][state];
    }


    /** <i>Greedy</i> */
    // 122. Best Time to Buy and Sell Stock II
    public int maxProfitII(int[] prices) {
        int profit = 0;
        if(prices.length == 0) return profit;
        for(int i=1;i<prices.length;i++){
            if(prices[i] > prices[i-1])
                profit += (prices[i] - prices[i-1]);
        }
        return profit;
    }
}
