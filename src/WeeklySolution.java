import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;

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
    /** Saturday - 6/12/2025 */
    public boolean completePrime(int num) {
        int n = 1;
        int a = 1;
        int b = 1;
        while (num >= n){
            a = num / n;
            if(a > 0 && notPrime(a)) return false;
            b = num - n * a;
            if(b > 0 && notPrime(b)) return false;
            n *= 10;
        }
        return true;
    }
    private boolean notPrime(int num){
        if(num < 2) return true;
        if(num == 2) return false;
        if(num % 2 == 0) return true;
        int n = (int)Math.sqrt(num);
        for(int i=3;i<=n;i+=2){
            if(num % i == 0) return true;
        }
        return false;
    }
    public int[] minOperations(int[] nums) {
        int num = 0;
        int[] res = new int[nums.length];
        for(int i=0;i<nums.length;i++){
            num = nums[i];
            for(int j=0;j<num;j++){
                if(isBinaryPalindrome(num + j)) {
                    res[i] = j;
                    break;
                }
                if(isBinaryPalindrome(num - j)){
                    res[i] = j;
                    break;
                }
            }
        }
        return res;
    }
    private boolean isBinaryPalindrome(int num){
        int count = 31;
        int l = 0;
        int r = 0;
        while ((num >> count) == 0){
            count -= 1;
        }
        int mid = (count + 1) / 2;
        int n = 0;
        while (n < mid){
            l = (num >> count) & 1;
            r = (num >> n) & 1;
            if(l != r) return false;
            count -= 1;
            n += 1;
        }
        return true;
    }
    public long maxPoints(int[] technique1, int[] technique2, int k) {
        int n = 0;
        long res = 0;
        int length = technique1.length;
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for(int i=0;i<length;i++){
            if(technique1[i] >= technique2[i]){
                res += technique1[i];
                n += 1;
            }
            else{
                res += technique2[i];
                pq.add(technique2[i] - technique1[i]);
            }
        }
        while (!pq.isEmpty() && n < k){
            res -= pq.poll();
            n += 1;
        }
        return res;
    }
}
