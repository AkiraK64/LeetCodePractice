import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class WeeklySolution {
    /** Sunday - 30/11/2025 */
    public int countElements(int[] nums, int k) {
        int length = nums.length;
        if(k == 0) return length;
        Arrays.sort(nums);
        int left = 0;
        int res = 0;
        for(int i=1;i<nums.length;i++){
            if(nums[i] == nums[i-1]) continue;
            if(length - i >= k) res += (i-left);
            left = i;
        }
        return res;
    }
    public int maxDistinct(String s) {
        HashSet<Character> set = new HashSet<>();
        for(var c : s.toCharArray()) set.add(c);
        return set.size();
    }
    public int minMirrorPairDistance(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int reversed = 0;
        int curNum = 0;
        int res = Integer.MAX_VALUE;
        for(int i=0;i<nums.length;i++){
            if(map.containsKey(nums[i])){
                res = Math.min(res, i - map.get(nums[i]));
            }
            else {
                reversed = 0;
                curNum = nums[i];
                while (curNum > 0){
                    reversed = reversed * 10 + curNum % 10;
                    curNum = curNum / 10;
                }
                map.put(reversed, i);
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }
    public long[] minOperations(int[] nums, int k, int[][] queries) {
        return new long[0];
    }
}
