import java.util.*;

/** <b>Neetcode 150</b>
 */
public class Solution {
    /** <i>Array And Hash</i>
     */
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
    // 374. Top K Frequent Elements
    public int[] topKFrequent(int[] nums, int k) {
        int length = nums.length;
        List<Integer>[] freq = new List[length+1];
        for(int i=0;i<=length;i++)
            freq[i] = new ArrayList<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for(var num : nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        for(var key : map.keySet())
            freq[map.get(key)].add(key);
        int index = 0;
        int[] res = new int[k];
        for(int i=length;i>=0;i--){
            if(freq[i].isEmpty()) continue;
            for(var e : freq[i]){
                res[index] = e;
                index += 1;
                if(index == k) return res;
            }
        }
        return res;
    }
    // 271. Encode and Decode Strings
    public String encode(List<String> strs) {
        StringBuilder res = new StringBuilder();
        for(var str : strs){
            res.append(str.length());
            res.append('#');
            res.append(str);
        }
        return res.toString();
    }
    public List<String> decode(String str) {
        int index = 0;
        int maxIndex = str.length() - 1;
        int length = 0;
        List<String> res = new ArrayList<>();
        char c = '#';
        while (index <= maxIndex){
            c = str.charAt(index);
            if(c == '#'){
                res.add(str.substring(index + 1, index + 1 + length));
                index += length;
                length = 0;
            }
            else{
                length = length * 10 + (c - '0');
            }
            index += 1;
        }
        return res;
    }
    // 238. Product Of Array Except Self
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] res = new int[length];
        int product = 1;
        for(int i=0;i<length;i++){
            res[i] = product;
            product *= nums[i];
        }
        product = 1;
        for (int i=length-1;i>=0;i--){
            res[i] *= product;
            product *= nums[i];
        }
        return res;
    }
    // 36. Valid Sudoku
    public boolean isValidSudoku(char[][] board) {
        List<int[]>[] map = new List[9];
        for(int i=0;i<9;i++) map[i] = new ArrayList<>();
        int value = 0;
        int rowIndex = 0;
        int colIndex = 0;
        for(var row : board){
            for(var col : row){
                if(col != '.') {
                    value = (col - '0') - 1;
                    for(var index : map[value]){
                        if(rowIndex == index[0]) return false;
                        if(colIndex == index[1]) return false;
                        if(rowIndex / 3 == index[0] / 3 && colIndex / 3 == index[1] / 3) return false;
                    }
                    map[value].add(new int[]{rowIndex, colIndex});
                }
                colIndex += 1;
            }
            rowIndex += 1;
            colIndex = 0;
        }
        return true;
    }
    // 128. Longest Consecutive Sequence
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for(var num : nums) numSet.add(num);
        int res = 0;
        for(var num : numSet){
            if(numSet.contains(num-1)) continue;
            int length = 1;
            while (numSet.contains(num + length))
                length += 1;
            res = Math.max(res, length);
        }
        return res;
    }


    /** <i>Two Pointers</i>
     */
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
    // 167. Two Sum II - Input Array Is Sorted
    public int[] twoSumSorted(int[] numbers, int target) {
        int leftIndex = 0;
        int rightIndex = numbers.length-1;
        int sum = 0;
        while (leftIndex < rightIndex){
            sum = numbers[leftIndex] + numbers[rightIndex];
            if(sum == target) return new int[]{leftIndex + 1, rightIndex + 1};
            else if(sum < target) leftIndex += 1;
            else rightIndex -= 1;
        }
        return new int[]{-1,-1};
    }
    // 15. 3sum
    public List<List<Integer>> threeSum(int[] nums) {
        int length = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        if(length < 3) return res;
        Arrays.sort(nums);
        int left = 0;
        int right = 0;
        int sum = 0;
        for(int i=0;i<length-2;i++){
            if(nums[i] > 0) break;
            if(i > 0 && nums[i] == nums[i-1]) continue;
            left = i+1;
            right = length-1;
            while (left < right){
                sum = nums[i] + nums[left] + nums[right];
                if(sum == 0){
                    res.add(List.of(nums[i], nums[left], nums[right]));
                    left += 1;
                    right -= 1;
                    while (left < right && nums[left] == nums[left-1])
                        left += 1;
                }
                else if(sum < 0){
                    left += 1;
                    while (left < right && nums[left] == nums[left-1])
                        left += 1;
                }
                else{
                    right -= 1;
                    while (left < right && nums[right] == nums[right+1])
                        right -= 1;
                }
            }
        }
        return res;
    }
    // 11. Container With Most Water
    public int maxArea(int[] height) {
        int length = height.length;
        int l = 0;
        int r = length - 1;
        int res = 0;
        while (l < r){
            res = Math.max(res, Math.min(height[l], height[r]) * (r - l));
            if(height[l] <= height[r]) l += 1;
            else r -= 1;
        }
        return res;
    }
    // 287. Find The Duplicate Number
    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (true){
            slow = nums[slow];
            fast = nums[nums[fast]];
            if(slow == fast) break;
        }
        int slow2 = 0;
        while (true){
            slow = nums[slow];
            slow2 = nums[slow2];
            if(slow == slow2) return slow;
        }
    }


    /** <i>Sliding Window</i>
     */
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
    // 424. Longest Repeating Character Replacement
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


    /** <i>Stack</i>
     */
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
    // 739. Daily Temperatures
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> tpStack = new Stack<>();
        tpStack.add(0);
        int prevDay = 0;
        int curTemp = 0;
        int prevTemp = 0;
        int[] res = new int[temperatures.length];
        for(int i=1;i<temperatures.length;i++){
            curTemp = temperatures[i];
            while (!tpStack.isEmpty()){
                prevDay = tpStack.peek();
                prevTemp = temperatures[prevDay];
                if(curTemp > prevTemp) {
                    res[prevDay] = i - prevDay;
                    tpStack.pop();
                }
                else break;
            }
            tpStack.add(i);
        }
        return res;
    }
    // 155. Min Stack
    // 150. Evaluate Reverse Polish Notation
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        HashSet<String> set = new HashSet<>(Set.of("+", "-", "*", "/"));
        int first = 0;
        int second = 0;
        for(var token : tokens){
            if(set.contains(token)) {
                second = stack.pop();
                first = stack.pop();
                if (Objects.equals(token, "+")) {
                    first = first + second;
                } else if (Objects.equals(token, "-")) {
                    first = first - second;
                } else if (Objects.equals(token, "*")) {
                    first = first * second;
                } else if (Objects.equals(token, "/")) {
                    first = first / second;
                }
            }
            else{
                first = Integer.parseInt(token);
            }
            stack.add(first);
        }
        return stack.pop();
    }
    // 853. Car Fleet
    public int carFleet(int target, int[] position, int[] speed) {
        int amount = position.length;
        int[][] cars = new int[amount][2];
        for(int i=0;i<amount;i++){
            cars[i][0] = position[i];
            cars[i][1] = speed[i];
        }
        Arrays.sort(cars, (x,y) -> (y[0] - x[0]));
        Stack<Double> carSt = new Stack<>();
        double time;
        for(int i=0;i<amount;i++){
            time = 1.0 * (target - cars[i][0]) / cars[i][1];
            if(carSt.isEmpty()) carSt.push(time);
            else if(time > carSt.peek()) carSt.push(time);
        }
        return carSt.size();
    }


    /** <i>Binary Search</i>
     */
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
    // 74. Search A 2D Matrix
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length;
        int col = matrix[0].length;
        int left = 0;
        int right = row * col - 1;
        int middle = 0;
        int rowIndex = 0;
        int colIndex = 0;
        while (left <= right){
            middle = (left + right) / 2;
            rowIndex = middle / col;
            colIndex = middle - rowIndex * col;
            if(matrix[rowIndex][colIndex] == target) return true;
            else if(matrix[rowIndex][colIndex] < target) left = middle + 1;
            else right = middle - 1;
        }
        return false;
    }
    // 875. Koko Eating Bananas
    public int minEatingSpeed(int[] piles, int h) {
        int l = 1;
        int r = 0;
        for(var pile : piles) if(pile > r) r = pile;
        int mid = 0;
        int res = r;
        int expectedHours = 0;
        boolean updated = true;
        while (l <= r){
            mid = (l + r) / 2;
            expectedHours = h;
            updated = true;
            for (var pile : piles) {
                expectedHours -= Math.ceilDiv(pile, mid);
                if(expectedHours < 0){
                    l = mid + 1;
                    updated = false;
                    break;
                }
            }
            if(updated){
                res = mid;
                r = mid - 1;
            }
        }
        return res;
    }
    // 153. Find Minimum in Rotated Sorted Array
    public int findMin(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        int res = Integer.MAX_VALUE;
        int mid = 0;
        while (l <= r){
            mid = (l + r) / 2;
            res = Math.min(res, nums[mid]);
            if(nums[mid] <= nums[l] && nums[mid] <= nums[r])
                r = mid - 1;
            else if(nums[mid] >= nums[l] && nums[mid] >= nums[r])
                l = mid + 1;
            else{
                res = Math.min(res, nums[l]);
                break;
            }
        }
        return res;
    }
    // 33. Search in Rotated Sorted Array
    public int searchII(int[] nums, int target) {
        int l = 0;
        int r = nums.length - 1;
        int mid = 0;
        while (l <= r){
            mid = (l + r) / 2;
            if(nums[mid] == target) return mid;
            else if(nums[mid] >= nums[l] && nums[mid] >= nums[r]){
                if(target < nums[mid] && target >= nums[l]) r = mid - 1;
                else l = mid + 1;
            }
            else if(nums[mid] <= nums[l] && nums[mid] <= nums[r]){
                if(target > nums[mid] && target <= nums[r]) l = mid + 1;
                else r = mid - 1;
            }
            else{
                if(target < nums[mid]) r = mid - 1;
                else l = mid + 1;
            }
        }
        return -1;
    }


    /** <i>Linked List</i>
     */
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
    // 141. Linked List Cycle (Floyd's Algorithm)
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
    // 143. Reorder List
    public void reorderList(ListNode head) {
        // FIND MID NODE
        var mid = head;
        var tail = head;
        var prevMid = new ListNode(0, mid);
        while (tail.next != null){
            prevMid = prevMid.next;
            mid = mid.next;
            tail = tail.next;
            if(tail.next != null) tail = tail.next;
        }
        // REVERSE NODE FROM MID NODE
        var curMid = mid;
        var nextMid = mid.next;
        while (nextMid != null){{
            prevMid = curMid;
            curMid = nextMid;
            nextMid = nextMid.next;
            curMid.next = prevMid;
        }}
        // RE-ORDER LIST
        var cur = head;
        var curNext = cur;
        var prevTail = tail;
        while (mid != tail){
            curNext = cur.next;
            prevTail = tail.next;
            cur.next = tail;
            tail.next = curNext;
            cur = curNext;
            tail = prevTail;
        }
        if(cur != mid) cur.next = tail;
        tail.next = null;
    }
    // 19. Remove Nth Node From End Of List
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int count = 0;
        var tail = head;
        var nthNode = head;
        while (tail.next != null && count < n){
            tail = tail.next;
            count += 1;
        }
        if(count == n-1) return head.next;
        else if(count < n-1) return head;
        while (tail.next != null){
            nthNode = nthNode.next;
            tail = tail.next;
        }
        if(nthNode.next != null) nthNode.next = nthNode.next.next;
        return head;
    }
    // 138. Copy List with Random Pointer
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        var newHead = new Node(head.val);
        var cur = head;
        var newCur = newHead;
        newCur.random = cur.random;
        Map<Node, Node> matchMap = new HashMap<>();
        matchMap.put(cur, newCur);
        while (cur.next != null){
            cur = cur.next;
            newCur.next = new Node(cur.val);
            newCur = newCur.next;
            newCur.random = cur.random;
            matchMap.put(cur, newCur);
        }
        newCur = newHead;
        while (newCur != null){
            if(newCur.random != null)
                newCur.random = matchMap.get(newCur.random);
            newCur = newCur.next;
        }
        return newHead;
    }
    // 2. Add Two Numbers
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int reminder = 0;
        int value = 0;
        var res = new ListNode(0, null);
        var head = res;
        while (l1 != null && l2 != null){
            value = l1.val + l2.val + reminder;
            reminder = value / 10;
            res.next = new ListNode(value % 10, null);
            res = res.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null){
            value = l1.val + reminder;
            reminder = value / 10;
            res.next = new ListNode(value % 10, null);
            res = res.next;
            l1 = l1.next;
        }
        while (l2 != null){
            value = l2.val + reminder;
            reminder = value / 10;
            res.next = new ListNode(value % 10, null);
            res = res.next;
            l2 = l2.next;
        }
        if(reminder != 0){
            res.next = new ListNode(reminder, null);
        }
        return head.next;
    }


    /** <i>Tree</i>
     */
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
    // 235. Lowest Common Ancestor Of Binary Search Tree
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || p == null || q == null)
            return null;
        if(Math.max(p.val, q.val) < root.val)
            return lowestCommonAncestor(root.left, p, q);
        else if(Math.min(p.val, q.val) > root.val)
            return lowestCommonAncestor(root.right, p, q);
        else return root;
    }
    // 102. Binary Tree Level Order Traversal
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> nodeQueue = new ArrayDeque<>();
        if(root != null) nodeQueue.add(root);
        boolean done = false;
        while (!done){
            var subRes = new ArrayList<Integer>();
            int size = nodeQueue.size();
            if(size == 0) done = true;
            else{
                while (size > 0){
                    var node = nodeQueue.poll();
                    if(node == null) continue;
                    subRes.add(node.val);
                    if(node.left != null) nodeQueue.add(node.left);
                    if(node.right != null) nodeQueue.add(node.right);
                    size -= 1;
                }
                res.add(subRes);
            }
        }
        return res;
    }
    // 199. Binary Tree Right Side View
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> dq = new ArrayDeque<>();
        if(root == null) return res;
        dq.offer(root);
        while (!dq.isEmpty()){
            res.add(dq.peekLast().val);
            int size = dq.size();
            TreeNode curNode;
            while (size > 0){
                curNode = dq.poll();
                if(curNode != null) {
                    if (curNode.left != null) dq.offer(curNode.left);
                    if (curNode.right != null) dq.offer(curNode.right);
                }
                size -= 1;
            }
        }
        return res;
    }
    // 1448. Count Good Nodes In Binary Tree
    public int goodNodes(TreeNode root) {
        return dfs_GoodNodes(root, root.val);
    }
    private int dfs_GoodNodes(TreeNode root, int maxVal){
        if(root == null)
            return 0;
        int res = 0;
        if(root.val >= maxVal) {
            res += 1;
            maxVal = root.val;
        }
        res += dfs_GoodNodes(root.left, maxVal);
        res += dfs_GoodNodes(root.right, maxVal);
        return res;
    }
    // 98. Validate Binary Search Tree
    public boolean isValidBST(TreeNode root) {
        return dfs_IsValidBST(root,null,null);
    }
    private boolean dfs_IsValidBST(TreeNode cur, TreeNode ancestorL, TreeNode ancestorR){
        if(cur == null) return true;
        if(ancestorL != null && cur.val >= ancestorL.val) return false;
        if(ancestorR != null && cur.val <= ancestorR.val) return false;
        return dfs_IsValidBST(cur.left, cur, ancestorR) && dfs_IsValidBST(cur.right, ancestorL, cur);
    }
    // 230. Kth Smallest Element In BTS
    public int kthSmallest(TreeNode root, int k) {
        int[] res = new int[]{k, 0};
        dfs_kthSmallest(root, res);
        return res[1];
    }
    private void dfs_kthSmallest(TreeNode root, int[] res){
        if(root == null) return;
        dfs_kthSmallest(root.left, res);
        if(res[0] == 0) return;
        res[0] -= 1;
        res[1] = root.val;
        dfs_kthSmallest(root.right, res);
    }
    // 105. Construct Binary Tree from Preorder and Inorder
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int[] index = new int[]{0,0};
        return dfs_BuildTree(preorder, inorder, Integer.MAX_VALUE, index);
    }
    private TreeNode dfs_BuildTree(int[] preorder, int[] inorder, int limit, int[] index){
        if(index[0] >= preorder.length) return null;
        if(inorder[index[1]] == limit){
            index[1] += 1;
            return null;
        }
        TreeNode node = new TreeNode(preorder[index[0]]);
        index[0] += 1;
        node.left = dfs_BuildTree(preorder, inorder, node.val, index);
        node.right = dfs_BuildTree(preorder, inorder, limit, index);
        return node;
    }


    /** <i>Heap & Priority Queue</i>
     */
    // 703. Kth Largest Element in a Stream
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
    // 215. Kth Largest Element in an Array
    public int findKthLargest(int[] nums, int k) {
        Queue<Integer> pq = new PriorityQueue<>();
        for(int num : nums){
            pq.add(num);
            if(pq.size() > k) pq.poll();
        }
        return pq.peek();
    }
    // 973. K Closest Points to Origin (DO IN PYTHON)
    // 621. Task Schedule
    public int leastInterval(char[] tasks, int n) {
        int[] fTask = new int[26];
        for(var task : tasks)
            fTask[task - 'A'] += 1;
        PriorityQueue<Integer> availTasks = new PriorityQueue<>(Collections.reverseOrder());
        for(int i=0;i<26;i++)
            if(fTask[i] > 0)
                availTasks.add(fTask[i]);
        Queue<int[]> cdTasks = new ArrayDeque<>();
        int time = 0;
        while (!availTasks.isEmpty() || !cdTasks.isEmpty()){
            time += 1;
            if(availTasks.isEmpty()){
                var nextTask = cdTasks.poll();
                time = nextTask[1];
                availTasks.add(nextTask[0]);
            }
            else if(!cdTasks.isEmpty() && time == cdTasks.peek()[1]){
                availTasks.add(cdTasks.poll()[0]);
            }
            var curTask = availTasks.poll();
            if(curTask > 1)
                cdTasks.add(new int[]{curTask-1, time + n + 1});
        }
        return time;
    }


    /** <i>Backtracking</i>
     */
    // 78. Subsets
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>((int)Math.pow(2, nums.length));
        ArrayList<Integer> subRes = new ArrayList<>(nums.length);
        bt_SubSets(-1, nums, subRes, res);
        return res;
    }
    private void bt_SubSets(int prev, int[] nums, ArrayList<Integer> subRes, List<List<Integer>> res){
        for(int i=prev; i<nums.length;i++){
            if(i == prev){
                res.add(new ArrayList<>(subRes));
            }
            else{
                subRes.add(nums[i]);
                bt_SubSets(i, nums, subRes, res);
                subRes.removeLast();
            }
        }
    }
    // 39. Combination Sum
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> subRes = new ArrayList<>(candidates.length);
        bt_CombinationSum(0, target, candidates, subRes, res);
        return res;
    }
    private void bt_CombinationSum(int index, int target, int[] candidates, ArrayList<Integer> subRes, List<List<Integer>> res){
        int left = 0;
        for(int i=index;i<candidates.length;i++){
            left = target - candidates[i];
            if(left < 0) continue;
            subRes.add(candidates[i]);
            if(left == 0) res.add(new ArrayList<>(subRes));
            else bt_CombinationSum(i, left, candidates, subRes, res);
            subRes.removeLast();
        }
    }
    // 40. Combination Sum II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> subRes = new ArrayList<>();
        Arrays.sort(candidates);
        bt_CombinationSum2(0, target, candidates, subRes, res);
        return res;
    }
    private void bt_CombinationSum2(int index, int target, int[] candidates, ArrayList<Integer> subRes, List<List<Integer>> res){
        int left = 0;
        for(int i=index;i<candidates.length;i++){
            if(i > index && candidates[i] == candidates[i-1]) continue;
            left = target - candidates[i];
            if(left < 0) continue;
            subRes.add(candidates[i]);
            if(left == 0) res.add(new ArrayList<>(subRes));
            else bt_CombinationSum2(i+1, left, candidates, subRes, res);
            subRes.removeLast();
        }
    }
    // 46. Permutations
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        ArrayList<Integer> subRes = new ArrayList<>();
        int n = nums.length;
        boolean[] used = new boolean[n];
        bt_Permute(0, n, nums, used, subRes, res);
        return res;
    }
    private void bt_Permute(int index, int n, int[] nums, boolean[] used, ArrayList<Integer> subRes, List<List<Integer>> res){
        for(int i=0;i<n;i++){
            if(!used[i]){
                used[i] = true;
                subRes.add(nums[i]);
                if(index == n-1) res.add(new ArrayList<>(subRes));
                else bt_Permute(index+1, n, nums, used, subRes, res);
                subRes.removeLast();
                used[i] = false;
            }
        }
    }
    // 51. N Queens
    public List<List<String>> solveNQueens(int n) {
        int[] subRes = new int[n];
        List<List<String>> res = new ArrayList<>();
        bt_SolveNQueens(0, n, subRes, res);
        return res;
    }
    private void bt_SolveNQueens(int index, int n, int[] subRes, List<List<String>> res){
        for(int i=0;i<n;i++){
            if(isNQueensCandidate(index, i, subRes)){
                subRes[index] = i;
                if(index == n-1){
                    List<String> rows = new ArrayList<>();
                    for(int j=0;j<n;j++){
                        StringBuilder row = new StringBuilder();
                        row.append(".".repeat(subRes[j]));
                        row.append('Q');
                        row.append(".".repeat(n-subRes[j]-1));
                        rows.add(row.toString());
                    }
                    res.add(rows);
                }
                else bt_SolveNQueens(index+1, n, subRes, res);
            }
        }
    }
    private boolean isNQueensCandidate(int row, int col, int[] subRes){
        for(int i=0;i<row;i++){
            if(subRes[i] == col) return false;
            if(Math.abs(subRes[i] - col) == row - i) return false;
        }
        return true;
    }


    /** <i>1-D Dynamic Programing</i>
     */
    // 70. Climbing Stairs
    public int climbStairs(int n) {
        int[] dp = new int[n+1];
        Arrays.fill(dp, -1);
        return bt_climbStairs(n, dp);
    }
    private int bt_climbStairs(int left, int[] dp){
        if(left < 0) return 0;
        if(left == 0) return 1;
        if(dp[left] != -1) return dp[left];
        dp[left] = bt_climbStairs(left - 1, dp) + bt_climbStairs(left - 2, dp);
        return dp[left];
    }
    // 746. Min Cost Climbing Stairs
    public int minCostClimbingStairs(int[] cost) {
        int step = cost.length + 1;
        int[] dp = new int[step + 1];
        Arrays.fill(dp, -1);
        return bt_minCostClimbingStairs(-1, step, cost, dp);
    }
    private int bt_minCostClimbingStairs(int index, int left, int[] costs, int[]dp){
        if(left == 0) return 0;
        if(dp[left] != -1) return dp[left];
        int cost = index == -1 ? 0 : costs[index];
        if(left == 1) dp[left] = cost;
        else{
            int costForOneStep = bt_minCostClimbingStairs(index + 1, left - 1, costs, dp);
            int costForTwoStep = bt_minCostClimbingStairs(index + 2, left - 2, costs, dp);
            dp[left] = cost + Math.min(costForOneStep, costForTwoStep);
        }
        return dp[left];
    }


    /** <i>2-D Dynamic Programing</i>
     */
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


    /** <i>Greedy</i>
     */
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


    /** <i>Intervals</i>
     */
    // 252. Meeting Rooms
    public boolean canAttendMeetings(List<Interval> intervals) {
        int length = intervals.size();
        if(length < 2) return true;

        intervals.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;
            }
        });

        for(int i=0;i<length-1;i++){
            if(intervals.get(i).end > intervals.get(i+1).start) return false;
        }
        return true;
    }


    /** <i>Math & Geometry</i>
     */
    // 202. Happy Number
    public boolean isHappy(int n) {
        if(n == 1) return true;
        HashSet<Integer> set = new HashSet<>();
        while (!set.contains(n)){
            set.add(n);
            n = sumOfSquares(n);
            if(n == 1) return true;
        }
        return false;
    }
    private int sumOfSquares(int n){
        int res = 0;
        int add = 0;
        while (n > 0) {
            add = n % 10;
            n = n / 10;
            res += add * add;
        }
        return res;
    }
    // 66. Plus One
    public int[] plusOne(int[] digits) {
        int length = digits.length;
        int[] res = new int[length+1];
        int value = 0;
        int reminder = 1;
        int index = res.length-1;
        for(int i=digits.length-1;i>=0;i--){
            if(reminder == 0) res[index] = digits[i];
            else {
                value = digits[i] + reminder;
                reminder = value / 10;
                res[index] = value % 10;
            }
            index -= 1;
        }
        if(reminder > 0) res[0] = reminder;
        else {
            System.arraycopy(res, 1, digits, 0, length);
            return digits;
        }
        return res;
    }
    // 50. Pow(x,n)
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        if(n == -1) return 1/x;
        if(n%2==0){
            double res = myPow(x, n/2);
            return res * res;
        }
        else{
            double res = myPow(x, (n-1)/2);
            return res * res * x;
        }
    }
    // 43. Multiply Strings
    public String multiply(String num1, String num2) {
        int num1_length = num1.length();
        int num2_length = num2.length();
        var num1_reverse = new StringBuilder(num1).reverse().toString();
        var num2_reverse = new StringBuilder(num2).reverse().toString();
        int[] res = new int[num1_length + num2_length];
        int digit1 = 0;
        int digit2 = 0;
        for(int i=0;i<num1_length;i++){
            for(int j=0;j<num2_length;j++){
                digit1 = num1_reverse.charAt(i) - '0';
                digit2 = num2_reverse.charAt(j) - '0';
                res[i+j] += digit1 * digit2;
                res[i+j+1] += res[i+j] / 10;
                res[i+j] = res[i+j] % 10;
            }
        }
        int endIndex = num1_length + num2_length - 1;
        while (endIndex >= 0 && res[endIndex] == 0)
            endIndex -= 1;
        if(endIndex == -1) return "0";
        StringBuilder resStr = new StringBuilder();
        while (endIndex >= 0){
            resStr.append(res[endIndex]);
            endIndex -= 1;
        }
        return resStr.toString();
    }
    // 48. Rotate Image
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        int half = n/2;
        for(int i=0;i<half;i++){
            for(int j=i;j<n-i-1;j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n-j-1][i];
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1];
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1];
                matrix[j][n-i-1] = tmp;
            }
        }
    }


    /** Bit Manipulation
     */
    // 136. Single Number
    public int singleNumber(int[] nums) {
        int res = 0;
        for(var num : nums){
            res ^= num;
        }
        return res;
    }
    // 191. Number of 1 Bits
    public int hammingWeight(int n) {
        int res = 0;
        while (n > 0){
            res += n & 1;
            n >>= 1;
        }
        return res;
    }
    // 338. Counting Bits
    public int[] countBits(int n) {
        int index = 1;
        int offset = 1;
        int[] res = new int[n+1];
        while (index <= n){
            if(index == 2 * offset){
                offset *= 2;
                res[index] = 1;
            }
            else {
                res[index] = res[index - offset] + 1;
            }
            index += 1;
        }
        return res;
    }
    public int[] countBits_SolutionTwo(int n){
        int[] res = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            res[i] = Integer.bitCount(i);
        }
        return res;
    }
    // 190. Reverse Bits
    public int reverseBits(int n) {
        int res = 0;
        int bit = 0;
        int count = 0;
        while (n > 0){
            bit = n & 1;
            n >>= 1;
            res <<= 1;
            res += bit;
            count += 1;
        }
        res <<= (32 - count);
        return res;
    }
    // 268. Missing Number
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int res = n;
        for(int i=0;i<n;i++){
            res ^= i;
            res ^= nums[i];
        }
        return res;
    }
    // 371. Sum Of Two Integers
    public int getSum(int a, int b) {
        int res = 0;
        int bitA = 0;
        int bitB = 0;
        int bitC = 0;
        int bitD = 0;
        int numBit = 0;
        int mask = 0xFFFFFFFF;
        while (numBit < 32){
            bitA = (a >> numBit) & 1;
            bitB = (b >> numBit) & 1;
            bitD = bitA ^ bitB ^ bitC;
            bitC = (bitA & bitB) | (bitA & bitC) | (bitB & bitC);
            if(bitD == 1) res |= (1 << numBit);
            numBit += 1;
        }
        if(res > mask) res = ~(res ^ mask);
        return res;
    }
    // 7. Reverse Integer
    public int reverse(int x) {
        final int MIN = Integer.MIN_VALUE;
        final int MAX = Integer.MAX_VALUE;
        final int sign = (x > 0 ? 1 : -1);
        if(x == MIN) return 0;
        int res = 0;
        int num = 0;
        x = x * sign;
        while (x != 0){
            num = x % 10;
            x = x / 10;
            if(res > (MAX - num) / 10) return 0;
            res = res * 10 + num;
        }
        return res * sign;
    }
}
