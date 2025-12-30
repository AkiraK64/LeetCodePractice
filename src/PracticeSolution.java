import com.sun.source.tree.Tree;

import java.util.*;

public class PracticeSolution {
    // 8. String to Integer (atoi)
    /** String */
    public int myAtoi(String s) {
        int sign = 1;
        int res = 0;
        boolean firstCharChecked = false;
        int MAX_VALUE = Integer.MAX_VALUE;
        int MIN_VALUE = Integer.MIN_VALUE;
        for(var c : s.toCharArray()){
            if(c == ' '){
                if(firstCharChecked) break;
                else continue;
            }
            if(c == '+' || c == '-'){
                if(firstCharChecked) break;
                else{
                    sign = c == '+' ? 1 : -1;
                    firstCharChecked = true;
                    continue;
                }
            }
            if(!firstCharChecked) firstCharChecked = true;
            int number = c - '0';
            if (number < 0 || number > 9) break;
            if (sign == 1 && (MAX_VALUE - number) / 10 < res) {
                res = MAX_VALUE;
                break;
            }
            else if (sign == -1 && res < (MIN_VALUE + number) / 10){
                res = MIN_VALUE;
                break;
            }
            else {
                res = res * 10 + sign * number;
            }
        }
        return res;
    }
    // 9. Palindrome Number
    /** String */
    public boolean isPalindrome(int x) {
        var str = String.valueOf(x);
        int length = str.length();
        int left = 0;
        int right = length-1;
        while (left < right){
            if(str.charAt(left) != str.charAt(right)) return false;
            left += 1;
            right -= 1;
        }
        return true;
    }
    // 13. Roman To Integer
    /** String */
    public int romanToInt(String s) {
        Map<Character, Integer> mapValue = new HashMap<>();
        Map<Character, Character> mapCase = new HashMap<>();
        mapValue.put('I', 1); mapValue.put('V', 5); mapValue.put('X', 10); mapValue.put('L', 50);
        mapValue.put('C', 100); mapValue.put('D', 500); mapValue.put('M', 1000);
        mapCase.put('V', 'I'); mapCase.put('X', 'I'); mapCase.put('L', 'X'); mapCase.put('C', 'X');
        mapCase.put('D', 'C'); mapCase.put('M', 'C');
        char prevChar = ' ';
        int res = 0;
        for(var c : s.toCharArray()){
            if(prevChar != ' ' && mapCase.getOrDefault(c, ' ') == prevChar){
                res -= 2 * mapValue.get(prevChar);
            }
            prevChar = c;
            res += mapValue.get(c);
        }
        return res;
    }
    // 108. Convert Sorted Array to Binary Search Tree
    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums.length == 0) return null;
        int middle = nums.length / 2;
        TreeNode root = new TreeNode(nums[middle]);
        var left = sortedArrayToBST(Arrays.copyOfRange(nums, 0, middle));
        var right = sortedArrayToBST(Arrays.copyOfRange(nums, middle+1, nums.length));
        root.left = left;
        root.right = right;
        return root;
    }
    // 109. Convert Sorted List to Binary Search Tree
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null) return null;
        if(head.next == null) return new TreeNode(head.val);
        var slow = head;
        var fast = head;
        var cur = head;
        while (fast.next != null){
            cur = slow;
            slow = slow.next;
            fast = fast.next;
            if(fast.next != null) fast = fast.next;
        }
        cur.next = null;
        var root = new TreeNode(slow.val);
        var left = sortedListToBST(head);
        var right = sortedListToBST(slow.next);
        root.left = left;
        root.right = right;
        return root;
    }
}
