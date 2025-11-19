import java.util.*;

public class SolutionOne {
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
    // 543. Diameter of Binary Tree
    public int diameterOfBinaryTree(TreeNode root) {
        int[] res = new int[1];
        dfs(root, res);
        return res[0];
    }
    private int dfs(TreeNode root, int[] res) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left, res);
        int right = dfs(root.right, res);
        res[0] = Math.max(res[0], left + right);
        return 1 + Math.max(left, right);
    }
    // 567. Permutation in String
    public boolean checkInclusion(String s1, String s2) {
        int slidePointer = -1;
        int s1_length = s1.length();
        if(s1_length == 0) return true;
        int s1_copyLength = s1_length;
        boolean[] charAppear = new boolean[26];
        int[] charCount = new int[26];
        for(var c : s1.toCharArray()){
            charAppear[c - 'a'] = true;
            charCount[c - 'a'] += 1;
        }
        int s2_length = s2.length();
        int curIndex = 0;
        int slideIndex = 0;
        for(int i=0;i<s2_length;i++){
            curIndex = s2.charAt(i) - 'a';
            if(!charAppear[curIndex]){
                if(slidePointer != -1) {
                    while (slidePointer < i){
                        slideIndex = s2.charAt(slidePointer) - 'a';
                        charCount[slideIndex] += 1;
                        slidePointer += 1;
                    }
                    slidePointer = -1;
                    s1_length = s1_copyLength;
                }
                continue;
            }
            if(charAppear[curIndex] && slidePointer == -1) slidePointer = i;
            while (charCount[curIndex] == 0){
                slideIndex = s2.charAt(slidePointer) - 'a';
                charCount[slideIndex] += 1;
                slidePointer += 1;
                s1_length += 1;
            }
            charCount[curIndex] -= 1;
            s1_length -= 1;
            if(s1_length == 0) return true;
        }
        return false;
    }
    // 494. Target Sum (Do 2/3)
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
    // 121. Best Time to Buy and Sell Stock (Do 2/3)
    public int maxProfit(int[] prices) {
        int l = 0;
        int r = 1;
        int profit = 0;
        int expectedProfit = 0;
        int days = prices.length;
        for(int i=0;i<days;i++){
            if(r>=days) break;
            expectedProfit = prices[r] - prices[l];
            if(expectedProfit < 0){
                l=i;
                r=i+1;
            }
            else{
                r++;
                if(profit < expectedProfit) profit = expectedProfit;
            }
        }
        return profit;
    }
    // 122. Best Time to Buy and Sell Stock II (Do 2/3)
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
