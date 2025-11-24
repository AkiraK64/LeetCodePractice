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
        int distance = 0;
        boolean found = false;
        for(var e : nums){
            if(e == 1){
                if (found){
                    if(distance < k) return false;
                    distance = 0;
                }
                else found = true;
            }
            else if(found){
                distance += 1;
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
}
